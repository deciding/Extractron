#import argparse
import os
#import subprocess
import time
import traceback
from datetime import datetime
import gc

import infolog
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from hparams import hparams_debug_string
from extractron.feeder import Feeder
from extractron.models import create_model
from extractron.utils import ValueWindow
from datasets.wavernn_audio import Audio
#from datasets import audio
import librosa
#from utils import plot
from utils.plotting import plot_spectrogram_to_numpy
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm
#import glob

log = infolog.log

def add_train_summary(summary_writer, step, loss):
    summary_writer.add_scalar('LOSS', loss, step)
    summary_writer.flush()

def add_train_stats(model, hparams):
    with tf.variable_scope('stats'):
        for i in range(hparams.extractron_num_gpus):
            tf.summary.histogram('predict_outputs %d' %
                                 i, model.tower_predict_outputs[i])
            tf.summary.histogram('target_spec %d' %
                                 i, model.tower_target_spec[i])
        tf.summary.scalar('before_loss', model.before_loss)
        tf.summary.scalar('after_loss', model.after_loss)


        tf.summary.scalar('regularization_loss', model.regularization_loss)
        tf.summary.scalar('loss', model.loss)
        # Control learning rate decay speed
        tf.summary.scalar('learning_rate', model.learning_rate)

        if hparams.extractron_teacher_forcing_mode == 'scheduled':
            # Control teacher forcing ratio decay when mode = 'scheduled'
            tf.summary.scalar('teacher_forcing_ratio', model.ratio)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        # visualize gradients (in case of explosion)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))

        return tf.summary.merge_all()

def add_eval_summary(summary_writer, step, before_loss, after_loss, loss,
        sample_rate, mixed_wav, target_wav, predicted_wav,
        mixed_spec_img, target_spec_img, predicted_spec_img):
    sdr = bss_eval_sources(target_wav, predicted_wav, False)[0][0]

    summary_writer.add_scalar('eval_before_loss', before_loss, step)
    summary_writer.add_scalar('eval_after_loss', after_loss, step)
    summary_writer.add_scalar('eval_loss', loss, step)
    summary_writer.add_scalar('SDR', sdr, step)

    summary_writer.add_audio('mixed_wav', mixed_wav, step, sample_rate)
    summary_writer.add_audio('target_wav', target_wav, step, sample_rate)
    summary_writer.add_audio('predicted_wav', predicted_wav, step, sample_rate)

    summary_writer.add_image('mixed_spectrogram', mixed_spec_img, step, dataformats='HWC')
    summary_writer.add_image('target_spectrogram', target_spec_img, step, dataformats='HWC')
    summary_writer.add_image('predicted_spectrogram', predicted_spec_img, step, dataformats='HWC')
    summary_writer.flush()

def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope('Extractron_model', reuse=tf.AUTO_REUSE):
        model_name = 'Extractron'
        model = create_model(model_name, hparams, args)
        model.initialize(feeder.mixed_spec, feeder.target_spec,
                spkid_embeddings=feeder.spkid_embeddings,
                global_step=global_step, is_training=True)
        model.add_loss(global_step)
        model.add_optimizer(global_step)
        stats = add_train_stats(model, hparams)
        return model, stats

def model_test_mode(args, feeder, hparams, global_step):
    with tf.variable_scope('Extractron_model', reuse=tf.AUTO_REUSE):
        model_name = 'Extractron'
        model = create_model(model_name, hparams)
        model.initialize(feeder.eval_mixed_spec, feeder.eval_target_spec,
                         feeder.eval_mixed_phase, feeder.eval_target_phase,
                         spkid_embeddings=feeder.eval_spkid_embeddings,
                         global_step=global_step, is_training=False)
        model.add_loss(global_step)
        return model

def train(log_dir, args, hparams):
    wavernn_audio=Audio(hparams)

    save_dir = os.path.join(log_dir, 'extract_pretrained')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    spec_dir = os.path.join(log_dir, 'spec-spectrograms')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    #eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    tensorboard_dir = os.path.join(log_dir, 'extractron_events')
    meta_folder = os.path.join(log_dir, 'metas')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    #os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'extractron_model.ckpt')
    checkpoint_path2 = os.path.join(save_dir, 'super_extractron_model.ckpt')
    #input_paths = [os.path.join(args.base_dir, args.extractron_input)]
    #if args.extractron_inputs:
    #    input_paths = [os.path.join(args.base_dir, arg_input_path)
    #                   for arg_input_path in args.extractron_inputs]
    #if args.extractron_input_glob:
    #    input_paths = glob.glob(args.extractron_input_glob)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.extractron_random_seed)

    # Set up data feeder
    with tf.variable_scope('datafeeder'):
        feeder = Feeder(hparams)
        feeder.setup_dataset(args.dataset, args.eval_dataset)
        class DotDict(dict):
            """
            a dictionary that supports dot notation
            as well as dictionary access notation
            usage: d = DotDict() or d = DotDict({'val1':'first'})
            set attributes: d.val2 = 'second' or d['val2'] = 'second'
            get attributes: d.val2 or d['val2']
            """
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

            def __init__(self, dct):
                for key, value in dct.items():
                    if hasattr(value, 'keys'):
                        value = DotDict(value)
                    self[key] = value
        dictkeys=['target_spec', 'mixed_spec', 'spkid_embeddings']
        eval_dictkeys=['eval_target_spec', 'eval_mixed_spec',
                'eval_target_phase', 'eval_mixed_phase','eval_spkid_embeddings']
        feeder_dict=DotDict(dict(zip(dictkeys, feeder.next)))
        feeder_dict.update(DotDict(dict(zip(eval_dictkeys, feeder.eval_next))))

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model, stats = model_train_mode(args, feeder_dict, hparams, global_step)
    eval_model = model_test_mode(args, feeder_dict, hparams, global_step)

    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5)
    saver2 = tf.train.Saver(max_to_keep=15)

    log('Extractron training set to a maximum of {} steps'.format(
        args.extractron_train_steps))

    # Memory allocation on the GPU as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement = True
    config.allow_soft_placement = True

    # Train
    with tf.Session(config=config) as sess:
        try:
            #summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            xsummary_writer = SummaryWriter(tensorboard_dir)

            sess.run(tf.global_variables_initializer())

            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)

                    if (checkpoint_state and checkpoint_state.model_checkpoint_path):
                        log('Loading checkpoint {}'.format(
                            checkpoint_state.model_checkpoint_path), slack=True)
                        saver.restore(
                            sess, checkpoint_state.model_checkpoint_path)

                    else:
                        log('No model to load at {}'.format(save_dir), slack=True)
                        saver.save(sess, checkpoint_path,
                                   global_step=global_step)

                except tf.errors.OutOfRangeError as e:
                    log('Cannot restore checkpoint: {}'.format(e), slack=True)
            else:
                log('Starting new training!', slack=True)
                saver.save(sess, checkpoint_path, global_step=global_step)

            if hparams.tfprof or hparams.timeline:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                if hparams.timeline:
                    from tensorflow.python.client import timeline
                if hparams.tfprof:
                    from tensorflow.python.profiler import model_analyzer, option_builder
                    my_profiler = model_analyzer.Profiler(graph=sess.graph)
                    profile_op_builder = option_builder.ProfileOptionBuilder( )
                    profile_op_builder.select(['micros', 'occurrence'])
                    profile_op_builder.order_by('micros')
                    #profile_op_builder.select(['device', 'bytes', 'peak_bytes'])
                    #profile_op_builder.order_by('bytes')
                    profile_op_builder.with_max_depth(20) # can be any large number
                    profile_op_builder.with_file_output('profile.log')
                    profile_op=profile_op_builder.build()

            # Training loop
            while step < args.extractron_train_steps:
                start_time = time.time()
                # from tensorflow.python import debug as tf_debug
                # sess=tf_debug.LocalCLIDebugWrapperSession(sess)
                if hparams.tfprof or hparams.timeline:
                    step, loss, opt = sess.run(
                        [global_step, model.loss, model.optimize], options=run_options, run_metadata=run_metadata)
                    if hparams.timeline:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format(show_dataflow=True, show_memory=True)
                        with open('timeline_01.json', 'w') as f:
                            f.write(chrome_trace)
                    if hparams.tfprof:
                        my_profiler.add_step(step=int(step), run_meta=run_metadata)
                        my_profiler.profile_name_scope(profile_op)
                else:
                    step, loss, opt = sess.run(
                        [global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = \
                'Step {:7d} [{:.3f} sec/step, {:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                    step, time.time() - start_time, time_window.average, loss, loss_window.average)

                log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

                # Originally assume 100 means loss exploded, now change to 1000 due to waveglow settings
                if loss > 100 or np.isnan(loss):
                    log('Loss exploded to {:.5f} at step {}'.format(
                        loss, step))
                    raise Exception('Loss exploded')

                if step % args.summary_interval == 0:
                    log('\nWriting summary at step {}'.format(step))
                    add_train_summary(xsummary_writer, step, loss)
                    #summary_writer.add_summary(sess.run(stats), step)
                    #summary_writer.flush()

                if step % args.gc_interval == 0:
                    log('\nGarbage collect: {}\n'.format(gc.collect()))

                if step % args.eval_interval == 0:
                    # Run eval and save eval stats
                    log('\nRunning evaluation at step {}'.format(step))

                    #1. avg loss, before, after, predicted mag, mixed phase, mixed_mag, target phase, target_mag
                    #2. 3 wavs
                    #3. 3 mag specs
                    #4. sdr

                    eval_losses = []
                    before_losses = []
                    after_losses = []

                    for i in tqdm(range(args.test_steps)):
                        try:
                            eloss, before_loss, after_loss, \
                            mixed_phase, mixed_spec, \
                            target_phase, target_spec, \
                            predicted_spec = sess.run([
                                eval_model.tower_loss[0], eval_model.tower_before_loss[0], eval_model.tower_after_loss[0],
                                eval_model.tower_mixed_phase[0][0], eval_model.tower_mixed_spec[0][0],
                                eval_model.tower_target_phase[0][0], eval_model.tower_target_spec[0][0],
                                eval_model.tower_predict_outputs[0][0]
                            ])
                            eval_losses.append(eloss)
                            before_losses.append(before_loss)
                            after_losses.append(after_loss)
                            #if i==0:
                            #    tmp_phase=mixed_phase
                            #    tmp_spec=mixed_spec
                        except tf.errors.OutOfRangeError:
                            log('\n test dataset out of range')
                            pass

                    eval_loss = sum(eval_losses) / len(eval_losses)
                    before_loss = sum(before_losses) / len(before_losses)
                    after_loss = sum(after_losses) / len(after_losses)

                    #mixed_wav = wavernn_audio.spec2wav(tmp_spec, tmp_phase)
                    mixed_wav = wavernn_audio.spec2wav(mixed_spec, mixed_phase)
                    target_wav = wavernn_audio.spec2wav(target_spec, target_phase)
                    predicted_wav = wavernn_audio.spec2wav(predicted_spec, mixed_phase)
                    librosa.output.write_wav(os.path.join(
                        eval_wav_dir, 'step-{}-eval-mixed.wav'.format(step)),
                        mixed_wav, hparams.sample_rate)
                    librosa.output.write_wav(os.path.join(
                        eval_wav_dir, 'step-{}-eval-target.wav'.format(step)),
                        target_wav, hparams.sample_rate)
                    librosa.output.write_wav(os.path.join(
                        eval_wav_dir, 'step-{}-eval-predicted.wav'.format(step)),
                        predicted_wav, hparams.sample_rate)
                    #audio.save_wav(mixed_wav, os.path.join(
                    #    eval_wav_dir, 'step-{}-eval-mixed.wav'.format(step)), sr=hparams.sample_rate)
                    #audio.save_wav(target_wav, os.path.join(
                    #    eval_wav_dir, 'step-{}-eval-target.wav'.format(step)), sr=hparams.sample_rate)
                    #audio.save_wav(predicted_wav, os.path.join(
                    #    eval_wav_dir, 'step-{}-eval-predicted.wav'.format(step)), sr=hparams.sample_rate)

                    mixed_spec_img=plot_spectrogram_to_numpy(mixed_spec.T)
                    target_spec_img=plot_spectrogram_to_numpy(target_spec.T)
                    predicted_spec_img=plot_spectrogram_to_numpy(predicted_spec.T)

                    #plot.plot_spectrogram(predicted_spec,
                    #        os.path.join(eval_plot_dir, 'step-{}-eval-spectrogram.png'.format(step)),
                    #        title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, eval_loss),
                    #        target_spectrogram=target_spec)

                    log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
                    log('Writing eval summary!')

                    add_eval_summary(xsummary_writer, step,
                                   before_loss, after_loss, eval_loss,
                                   hparams.sample_rate, mixed_wav, target_wav, predicted_wav,
                                   mixed_spec_img, target_spec_img, predicted_spec_img)

                if step % args.super_checkpoint_interval == 0 or step == args.extractron_train_steps:
                    # Save model and current global step
                    saver2.save(sess, checkpoint_path2, global_step=global_step)

                if step % args.checkpoint_interval == 0 or step == args.extractron_train_steps:
                    # Save model and current global step
                    saver.save(sess, checkpoint_path, global_step=global_step)

                    #log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')

                    #input_seq, mel_prediction, alignment, target, target_length = sess.run([
                    #    model.tower_inputs[0][0],
                    #    model.tower_mel_outputs[0][0],
                    #    model.tower_alignments[0][0],
                    #    model.tower_mel_targets[0][0],
                    #    model.tower_targets_lengths[0][0],
                    #])

                    ## save predicted mel spectrogram to disk (debug)
                    #mel_filename = 'mel-prediction-step-{}.npy'.format(step)
                    #np.save(os.path.join(mel_dir, mel_filename),
                    #        mel_prediction.T, allow_pickle=False)

                    ## save griffin lim inverted wav for debug (mel -> wav)
                    #wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                    #audio.save_wav(wav, os.path.join(
                    #    wav_dir, 'step-{}-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)

                    ## save alignment plot to disk (control purposes)
                    #plot.plot_alignment(alignment, os.path.join(plot_dir, 'step-{}-align.png'.format(step)),
                    #                    title='{}, {}, step={}, loss={:.5f}'.format(
                    #                        args.model, time_string(), step, loss),
                    #                    max_len=target_length // hparams.outputs_per_step)
                    ## save real and predicted mel-spectrogram plot to disk (control purposes)
                    #plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir, 'step-{}-mel-spectrogram.png'.format(step)),
                    #                      title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step, loss), target_spectrogram=target,
                    #                      max_len=target_length)
                    #log('Input at step {}: {}'.format(
                    #    step, sequence_to_text(input_seq)))


            log('Extractron training complete after {} global steps!'.format(
                args.extractron_train_steps), slack=True)
            return save_dir

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()

def extractron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)
