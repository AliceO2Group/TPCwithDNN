# pylint: disable=too-many-instance-attributes, too-many-statements, too-many-arguments, fixme
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
import os
from array import array
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from root_numpy import fill_hist
from ROOT import TH1F, TH2F, TFile, TCanvas, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT, TTree  # pylint: disable=import-error, no-name-in-module
from symmetry_padding_3d import SymmetryPadding3d
from machine_learning_hep.logger import get_logger
from fluctuation_data_generator import FluctuationDataGenerator
from utilities_dnn import u_net
from data_loader import load_train_apply, load_data_original, get_event_mean_indices

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use("Agg")

class DnnOptimiser:
    # Class Attribute
    # TODO: What is this for?
    species = "dnnoptimiser"

    h_dist_name = "h_dist"
    h_deltas_name = "h_deltas"
    h_deltas_vs_dist_name = "h_deltas_vs_dist"
    profile_name = "profile_deltas_vs_dist"
    h_std_dev_name = "h_std_dev"

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DnnOptimizer::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.selopt_input = data_param["selopt_input"]
        self.selopt_output = data_param["selopt_output"]
        self.opt_train = data_param["opt_train"]
        self.opt_predout = data_param["opt_predout"]
        self.nameopt_predout = data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.use_scaler = data_param["use_scaler"]

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirval = data_param["dirval"]
        self.diroutflattree = data_param["diroutflattree"]
        self.dirinput_train = data_param["dirinput_train"] + "/SC-%d-%d-%d/" % \
                (self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = data_param["dirinput_apply"] + "/SC-%d-%d-%d/" % \
                (self.grid_z, self.grid_r, self.grid_phi)

        # DNN config
        self.filters = data_param["filters"]
        self.pooling = data_param["pooling"]
        self.batch_size = data_param["batch_size"]
        self.shuffle = data_param["shuffle"]
        self.depth = data_param["depth"]
        self.batch_normalization = data_param["batch_normalization"]
        self.dropout = data_param["dropout"]
        self.epochs = data_param["epochs"]
        self.lossfun = data_param["lossfun"]
        self.metrics = data_param["metrics"]
        self.adamlr = data_param["adamlr"]

        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'selopt_input' : self.selopt_input,
                       'selopt_output' : self.selopt_output,
                       'data_dir': self.dirinput_train,
                       'use_scaler': self.use_scaler}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization, self.use_scaler)
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        if not os.path.isdir("plots"):
            os.makedirs("plots")

        if not os.path.isdir(self.dirmodel):
            os.makedirs(self.dirmodel)

        if not os.path.isdir(self.dirval):
            os.makedirs(self.dirval)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        self.indices_events_means_train, self.partition = get_event_mean_indices(
            data_param["maxrandomfiles_train"], data_param["maxrandomfiles_apply"],
            data_param['range_mean_index'], data_param['rangeevent_train'],
            data_param['rangeevent_test'], data_param['rangeevent_apply'])

        gROOT.SetStyle("Plain")
        gROOT.SetBatch()


    # pylint: disable=too-many-locals
    def dumpflattree(self):
        self.logger.info("DnnOptimizer::dumpflattree")
        self.logger.warning("DO YOU REALLY WANT TO DO IT? IT TAKES TIME")
        outfile_name = "%s/tree%s.root" % (self.diroutflattree, self.suffix_ds)
        myfile = TFile.Open(outfile_name, "recreate")

        tree = TTree('tvoxels', 'tree with histos')
        indexr = array('i', [0])
        indexphi = array('i', [0])
        indexz = array('i', [0])
        posr = array('f', [0])
        posphi = array('f', [0])
        posz = array('f', [0])
        evtid = array('i', [0])
        meanid = array('i', [0])
        randomid = array('i', [0])
        distmeanr = array('f', [0])
        distmeanrphi = array('f', [0])
        distmeanz = array('f', [0])
        distrndr = array('f', [0])
        distrndrphi = array('f', [0])
        distrndz = array('f', [0])
        tree.Branch('indexr', indexr, 'indexr/I')
        tree.Branch('indexphi', indexphi, 'indexphi/I')
        tree.Branch('indexz', indexz, 'indexz/I')
        tree.Branch('posr', posr, 'posr/F')
        tree.Branch('posphi', posphi, 'posphi/F')
        tree.Branch('posz', posz, 'posz/F')
        tree.Branch('distmeanr', distmeanr, 'distmeanr/F')
        tree.Branch('distmeanrphi', distmeanrphi, 'distmeanrphi/F')
        tree.Branch('distmeanz', distmeanz, 'distmeanz/F')
        tree.Branch('distrndr', distrndr, 'distrndr/F')
        tree.Branch('distrndrphi', distrndrphi, 'distrndrphi/F')
        tree.Branch('distrndz', distrndz, 'distrndz/F')
        tree.Branch('evtid', evtid, 'evtid/I')
        tree.Branch('meanid', meanid, 'meanid/I')
        tree.Branch('randomid', randomid, 'randomid/I')

        for indexev in self.indices_events_means_train:
            self.logger.info("processing event: %d", indexev)

            # TODO: Should it be for train or apply data?
            [vec_r_pos, vec_phi_pos, vec_z_pos,
             _, _,
             vec_mean_dist_r, vec_random_dist_r,
             vec_mean_dist_rphi, vec_random_dist_rphi,
             vec_mean_dist_z, vec_random_dist_z] = load_data_original(self.dirinput_train, indexev)

            vec_r_pos = vec_r_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_phi_pos = vec_phi_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_z_pos = vec_z_pos.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_r = vec_mean_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_r = vec_random_dist_r.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_mean_dist_rphi = vec_mean_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                            self.grid_z*2)
            vec_random_dist_rphi = vec_random_dist_rphi.reshape(self.grid_phi, self.grid_r,
                                                                self.grid_z*2)
            vec_mean_dist_z = vec_mean_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)
            vec_random_dist_z = vec_random_dist_z.reshape(self.grid_phi, self.grid_r, self.grid_z*2)

            for indexphi in range(self.grid_phi):
                for indexr in range(self.grid_r):
                    for indexz in range(self.grid_z*2):
                        indexphi[0] = indexphi
                        indexr[0] = indexr
                        indexz[0] = indexz
                        posr[0] = vec_r_pos[indexphi][indexr][indexz]
                        posphi[0] = vec_phi_pos[indexphi][indexr][indexz]
                        posz[0] = vec_z_pos[indexphi][indexr][indexz]
                        distmeanr[0] = vec_mean_dist_r[indexphi][indexr][indexz]
                        distmeanrphi[0] = vec_mean_dist_rphi[indexphi][indexr][indexz]
                        distmeanz[0] = vec_mean_dist_z[indexphi][indexr][indexz]
                        distrndr[0] = vec_random_dist_r[indexphi][indexr][indexz]
                        distrndrphi[0] = vec_random_dist_rphi[indexphi][indexr][indexz]
                        distrndz[0] = vec_random_dist_z[indexphi][indexr][indexz]
                        evtid[0] = indexev[0] + 10000*indexev[1]
                        meanid[0] = indexev[1]
                        randomid[0] = indexev[0]
                        tree.Fill()
        myfile.Write()
        myfile.Close()
        self.logger.info("Tree written in %s", outfile_name)


    def train(self):
        self.logger.info("DnnOptimizer::train")

        training_generator = FluctuationDataGenerator(self.partition['train'], **self.params)
        validation_generator = FluctuationDataGenerator(self.partition['validation'], **self.params)
        model = u_net((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),
                      depth=self.depth, batchnorm=self.batch_normalization,
                      pool_type=self.pooling, start_channels=self.filters, dropout=self.dropout)
        model.compile(loss=self.lossfun, optimizer=Adam(lr=self.adamlr),
                      metrics=[self.metrics]) # Mean squared error

        model.summary()
        plot_model(model, to_file='plots/model%s.png' % (self.suffix),
                   show_shapes=True, show_layer_names=True)

        his = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  use_multiprocessing=True,
                                  epochs=self.epochs, workers=1)

        plt.style.use("ggplot")
        plt.figure()
        plt.yscale('log')
        plt.plot(np.arange(0, self.epochs), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), his.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), his.history[self.metrics],
                 label="train_" + self.metrics)
        plt.plot(np.arange(0, self.epochs), his.history["val_" + self.metrics],
                 label="val_" + self.metrics)
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plots/plot_%s.png" % self.suffix)

        model_json = model.to_json()
        with open("%s/model%s.json" % (self.dirmodel, self.suffix), "w") as json_file: \
            json_file.write(model_json)
        model.save_weights("%s/model%s.h5" % (self.dirmodel, self.suffix))
        self.logger.info("Saved trained model to disk")

    # TODO: What is it for? To remove?
    def groupbyindices_input(self, arrayflat):
        return arrayflat.reshape(1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input)

    def apply(self):
        self.logger.info("DnnOptimizer::apply, input size: %d", self.dim_input)

        json_file = open("%s/model%s.json" % (self.dirmodel, self.suffix), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = \
            model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
        loaded_model.load_weights("%s/model%s.h5" % (self.dirmodel, self.suffix))

        myfile = TFile.Open("%s/output%s.root" % (self.dirval, self.suffix), "recreate")
        h_dist_all_events = TH2F("%s_all_events_%s" % (self.h_dist_name, self.suffix), "",
                                 500, -5, 5, 500, -5, 5)
        h_deltas_all_events = TH1F("%s_all_events_%s" % (self.h_deltas_name, self.suffix), "",
                                   1000, -1., 1.)
        h_deltas_vs_dist_all_events = TH2F("%s_all_events_%s" % \
                                           (self.h_deltas_vs_dist_name, self.suffix), "",
                                           500, -5.0, 5.0, 100, -0.5, 0.5)

        for iexperiment in self.partition['apply']:
            indexev = iexperiment
            inputs_, exp_outputs_ = load_train_apply(self.dirinput_apply, indexev,
                                                     self.selopt_input, self.selopt_output,
                                                     self.grid_r, self.grid_phi, self.grid_z,
                                                     self.opt_train, self.opt_predout)
            inputs_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input))
            exp_outputs_single = np.empty((1, self.grid_phi, self.grid_r,
                                           self.grid_z, self.dim_output))
            inputs_single[0, :, :, :, :] = inputs_
            exp_outputs_single[0, :, :, :, :] = exp_outputs_

            distortion_predict_group = loaded_model.predict(inputs_single)
            distortion_predict_flat_m = distortion_predict_group.reshape(-1, 1)
            distortion_predict_flat_a = distortion_predict_group.flatten()

            distortion_numeric_group = exp_outputs_single
            distortion_numeric_flat_m = distortion_numeric_group.reshape(-1, 1)
            distortion_numeric_flat_a = distortion_numeric_group.flatten()
            deltas_flat_a = (distortion_predict_flat_a - distortion_numeric_flat_a)
            deltas_flat_m = (distortion_predict_flat_m - distortion_numeric_flat_m)

            h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
            h_dist = TH2F("%s_%s" % (self.h_dist_name, h_suffix), "", 500, -5, 5, 500, -5, 5)
            h_deltas = TH1F("%s_%s" % (self.h_deltas_name, h_suffix), "", 1000, -1., 1.)
            h_deltas_vs_dist = TH2F("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix), "",
                                    500, -5.0, 5.0, 100, -0.5, 0.5)

            fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                distortion_predict_flat_m), axis=1))
            fill_hist(h_dist, np.concatenate((distortion_numeric_flat_m,
                                              distortion_predict_flat_m), axis=1))
            fill_hist(h_deltas, deltas_flat_a)
            fill_hist(h_deltas_all_events, deltas_flat_a)
            fill_hist(h_deltas_vs_dist,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))
            fill_hist(h_deltas_vs_dist_all_events,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

            prof = h_deltas_vs_dist.ProfileX()
            prof.SetName("%s_%s" % (self.profile_name, h_suffix))

            h_dist.Write()
            h_deltas.Write()
            h_deltas_vs_dist.Write()
            prof.Write()

            h1tmp = h_deltas_vs_dist.ProjectionX("h1tmp")
            h_std_dev = h1tmp.Clone("%s_%s" % (self.h_std_dev_name, h_suffix))
            h_std_dev.Reset()
            h_std_dev.SetXTitle("Numerical distortion fluctuation (cm)")
            h_std_dev.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
            nbin = int(h_std_dev.GetNbinsX())
            for ibin in range(0, nbin):
                h1diff = h_deltas_vs_dist.ProjectionY("h1diff", ibin+1, ibin+1, "")
                stddev = h1diff.GetStdDev()
                stddev_err = h1diff.GetStdDevError()
                h_std_dev.SetBinContent(ibin+1, stddev)
                h_std_dev.SetBinError(ibin+1, stddev_err)
            h_std_dev.Write()

        h_dist_all_events.Write()
        h_deltas_all_events.Write()
        h_deltas_vs_dist_all_events.Write()
        prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
        prof_all_events.SetName("%s_all_events_%s" % (self.profile_name, self.suffix))
        prof_all_events.Write()

        h1tmp = h_deltas_vs_dist_all_events.ProjectionX("h1tmp")
        h_std_dev_all_events = h1tmp.Clone("%s_all_events_%s" % (self.h_std_dev_name, self.suffix))
        h_std_dev_all_events.Reset()
        h_std_dev_all_events.SetXTitle("Numerical distortion fluctuation (cm)")
        h_std_dev_all_events.SetYTitle("std.dev. of (Pred. - Num.) distortion fluctuation (cm)")
        nbin = int(h_std_dev_all_events.GetNbinsX())
        for ibin in range(0, nbin):
            h1diff = h_deltas_vs_dist_all_events.ProjectionY("h1diff", ibin+1, ibin+1, "")
            stddev = h1diff.GetStdDev()
            stddev_err = h1diff.GetStdDevError()
            h_std_dev_all_events.SetBinContent(ibin+1, stddev)
            h_std_dev_all_events.SetBinError(ibin+1, stddev_err)
        h_std_dev_all_events.Write()

        myfile.Close()
        self.logger.info("Done apply")


    @staticmethod
    def plot_distorsion(h_dist, h_deltas, h_deltas_vs_dist, prof, suffix, opt_name):
        cev = TCanvas("canvas_%s_%s" % (suffix, opt_name), "canvas_%s_%s" % (suffix, opt_name),
                      1400, 1000)
        cev.Divide(2, 2)
        cev.cd(1)
        h_dist.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % opt_name)
        h_dist.GetYaxis().SetTitle("Predicted distortion fluctuation (cm)")
        h_dist.Draw("colz")
        cev.cd(2)
        gPad.SetLogy()
        h_deltas_vs_dist.GetXaxis().SetTitle("Numeric %s distorsion fluctuation (cm)" % opt_name)
        h_deltas_vs_dist.ProjectionX().Draw()
        h_deltas_vs_dist.GetYaxis().SetTitle("Entries")
        cev.cd(3)
        gPad.SetLogy()
        h_deltas.GetXaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)"
                                     % opt_name)
        h_deltas.GetYaxis().SetTitle("Entries")
        h_deltas.Draw()
        cev.cd(4)
        prof.GetYaxis().SetTitle("(Predicted - Numeric) %s distortion fluctuation (cm)" % opt_name)
        prof.GetXaxis().SetTitle("Numeric %s distortion fluctuation (cm)" % opt_name)
        prof.Draw()
        #cev.cd(5)
        #h_deltas_vs_dist.GetXaxis().SetTitle("Numeric R distorsion (cm)")
        #h_deltas_vs_dist.GetYaxis().SetTitle("(Predicted - Numeric) R distorsion (cm)")
        #h_deltas_vs_dist.Draw("colz")
        cev.SaveAs("plots/canvas_%s.pdf" % (suffix))

    def plot(self):
        self.logger.info("DnnOptimizer::plot")
        for iname, opt in enumerate(self.opt_predout):
            if opt == 1:
                opt_name = self.nameopt_predout[iname]

                myfile = TFile.Open("%s/output%s.root" % (self.dirval, self.suffix), "open")
                h_dist_all_events = myfile.Get("%s_all_events_%s" % (self.h_dist_name, self.suffix))
                h_deltas_all_events = myfile.Get("%s_all_events_%s" % \
                                                 (self.h_deltas_name, self.suffix))
                h_deltas_vs_dist_all_events = myfile.Get("%s_all_events_%s" % \
                                                         (self.h_deltas_vs_dist_name, self.suffix))
                profile_deltas_vs_dist_all_events = \
                    myfile.Get("%s_all_events_%s" % (self.profile_name, self.suffix))
                self.plot_distorsion(h_dist_all_events, h_deltas_all_events,
                                     h_deltas_vs_dist_all_events, profile_deltas_vs_dist_all_events,
                                     self.suffix, opt_name)

                counter = 0
                for iexperiment in self.partition['apply']:
                    h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
                    h_dist = myfile.Get("%s_%s" % (self.h_dist_name, h_suffix))
                    h_deltas = myfile.Get("%s_%s" % (self.h_deltas_name, h_suffix))
                    h_deltas_vs_dist = myfile.Get("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix))
                    profile = myfile.Get("%s_%s" % (self.profile_name, h_suffix))
                    self.plot_distorsion(h_dist, h_deltas, h_deltas_vs_dist, profile,
                                         h_suffix, opt_name)
                    counter = counter + 1
                    if counter > 100:
                        return


    # pylint: disable=no-self-use
    def gridsearch(self):
        self.logger.warning("Grid search not yet implemented")
