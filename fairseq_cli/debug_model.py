#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import logging
import math
import os
import random
import sys
import json

import numpy as np
import torch

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.trainer import Trainer
from fairseq.model_parallel.megatron_trainer import MegatronTrainer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.train')


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu and not getattr(args, 'tpu', False):
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # # Load valid dataset (we load training data below, based on the latest checkpoint)
    # for valid_sub_split in args.valid_subset.split(','):
    #     task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info('model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer=None)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info('training on {} devices (GPUs/TPUs)'.format(args.distributed_world_size))
    logger.info('max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    checkpoint_dir = '/raj-learn/checkpoints/lr-rewind_0.75sparsity_0.2frac_30epochs/'
    files = ['checkpoint_LTH0_epoch60.pt',
             'checkpoint_LTH1_epoch60_sparsity0.168.pt',
             'checkpoint_LTH2_epoch60_sparsity0.302.pt',
             'checkpoint_LTH3_epoch60_sparsity0.410.pt',
             'checkpoint_LTH4_epoch60_sparsity0.496.pt',
             'checkpoint_LTH5_epoch60_sparsity0.565.pt',
             'checkpoint_LTH6_epoch60_sparsity0.620.pt',
             'checkpoint_LTH7_epoch60_sparsity0.664.pt',
             'checkpoint_LTH8_epoch60_sparsity0.699.pt',
             ]

    # fn = 'checkpoint_LTH0_epoch60.pt'
    # trainer.load_checkpoint(os.path.join(checkpoint_dir, fn))
    # model = trainer.get_model()


    for fn in files:
        trainer.load_checkpoint(os.path.join(checkpoint_dir, fn))
        print("Mask sparsity")
        print(trainer.get_model().get_sparsity())
        print("Manual sparsity")
        print(trainer.get_model().get_manual_sparsity())
        
        # sparsities = trainer.get_model().get_layerwise_sparsity()
        # with open(f'./analysis/layer_sparsities/layerSparsities_{fn}.json', 'w') as outfile:
        #    json.dump(sparsities, outfile)

    # trainer.load_checkpoint(os.path.join(checkpoint_dir, 'checkpoint_LTH0_epoch60.pt'))
    # trainer.get_model().random_prune(0.2)
    # trainer.get_model().apply_masks()
    # print(trainer.get_model().get_sparsity())
    # print(trainer.get_model().get_manual_sparsity())
    # trainer.get_model().random_prune(0.2)
    # trainer.get_model().apply_masks()
    # print(trainer.get_model().get_sparsity())
    # print(trainer.get_model().get_manual_sparsity())

    print("main() complete")


def iterative_pruning_and_rewinding(args, task, trainer):
    # p = 1 - s^(1/n)
    prune_frac = 1 - (1 - args.final_sparsity)**(1/args.n_lth_iterations)

    # store the initial mask
    cur_mask = trainer.get_model().get_masks()
    max_epoch = args.max_epoch or math.inf
    for itr in range(args.n_lth_iterations):
        logger.info('IMP training iteration {}; current sparsity: {}'.format(
            itr,
            trainer.get_model().get_sparsity()
        ))
        # On first LTH iteration, load from latest checkpoint if available
        if itr == 0:
            extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
        if itr != 0:
            rewind_checkpoint = os.path.join(args.save_dir, 
                                             'checkpoint10.pt')
                                             # f'checkpoint_LTH{itr-1}_iter{args.lth_rewind_iter}.pt')
            trainer.load_checkpoint(rewind_checkpoint,
                                    reset_optimizer=False,
                                    reset_lr_scheduler=False
                                    )

            # set the rewinded model's mask to current mask, and apply the mask
            trainer.get_model().set_masks(cur_mask)
            trainer.get_model().apply_masks()
            epoch_itr = trainer.get_train_iterator(epoch=1, load_dataset=True)

        lr = trainer.get_lr()
        train_meter = meters.StopwatchMeter()
        train_meter.start()
        while (
            lr > args.min_lr
            and epoch_itr.next_epoch_idx <= max_epoch
        ):
            # train for one epoch
            valid_losses, should_stop = train(args, trainer, task, epoch_itr)
            if should_stop:
                break

            # only use first validation loss to update the learning rate
            lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=(os.pathsep in getattr(args, 'data', '')),
            )
        train_meter.stop()
        logger.info('done training IMP iteration {} in {:.1f} seconds'.format(itr, train_meter.sum))

        # save this model
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, None, 
                                         custom_filename=f'checkpoint_LTH{itr}_epoch{epoch_itr.epoch}.pt')

        # update mask by pruning, and store new mask
        trainer.get_model().prune_weights(prune_frac)
        cur_mask = trainer.get_model().get_masks()


def learning_rate_rewinding(args, task, trainer):
    # p = 1 - s^(1/n)
    prune_frac = 1 - (1 - args.final_sparsity)**(1/args.n_lth_iterations)

    max_epoch = args.max_epoch or math.inf
    for lth_iter in range(args.n_lth_iterations):
        logger.info('IMP training iteration {}; current sparsity: {}'.format(
            lth_iter,
            trainer.get_model().get_sparsity()
        ))
        # On first LTH iteration, load from latest checkpoint if available
        if lth_iter == 0:
            extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
        if lth_iter != 0:
            # rewind the model's update count to halfway through training
            # this will also set the learning rate to the value at halfway through train
            cur_update_count = trainer.get_num_updates()
            rewind_update = int(0.5*cur_update_count)
            trainer.set_num_updates(rewind_update)
            logger.info('Rewound update count to %d' % rewind_update)
            
            # set epoch number to halfway point
            reset_epoch = max_epoch // 2 if max_epoch is not math.inf else 1
            epoch_itr = trainer.get_train_iterator(epoch=reset_epoch, load_dataset=True)

        lr = trainer.get_lr()
        train_meter = meters.StopwatchMeter()
        train_meter.start()
        while (
            lr > args.min_lr
            and epoch_itr.next_epoch_idx <= max_epoch
        ):
            logger.info('epoch {}; current sparsity: {}'.format(
                epoch_itr.epoch,
                trainer.get_model().get_manual_sparsity()
            ))
            # train for one epoch
            valid_losses, should_stop = train(args, trainer, task, epoch_itr)
            if should_stop:
                break

            # only use first validation loss to update the learning rate
            lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                # sharded data: get train iterator for next epoch
                load_dataset=(os.pathsep in getattr(args, 'data', '')),
            )
        train_meter.stop()
        logger.info('done training IMP iteration {} in {:.1f} seconds'.format(lth_iter, train_meter.sum))

        # save this model
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, None, 
                                         custom_filename=f'checkpoint_LTH{lth_iter}_epoch{epoch_itr.epoch}.pt')

        # update mask by pruning, and apply mask
        trainer.get_model().prune_weights(prune_frac)
        trainer.get_model().apply_masks()


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info('early stop since valid performance hasn\'t improved for last {} runs'.format(args.patience))
            return True
        else:
            return False


def tpu_data_loader(args, itr):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    xm.rendezvous('tpu_data_loader')  # wait for all workers
    xm.mark_step()
    device = utils.get_tpu_device(args)
    return iterators.CountingIterator(
        pl.ParallelLoader(itr, [device]).per_device_loader(device),
        start=getattr(itr, 'n', 0),
        total=len(itr),
    )


@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, 'tpu', False):
        itr = tpu_data_loader(args, itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
    )

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = args.valid_subset.split(',')
    should_stop = False
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:  # OOM, overflow, ...
                continue

        # log mid-epoch stats
        num_updates = trainer.get_num_updates()
        if num_updates % args.log_interval == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)

            # reset mid-epoch stats after each log interval
            # the end-of-epoch stats will still be preserved
            metrics.reset_meters('train_inner')

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )
        if should_stop:
            break

    # log end-of-epoch stats
    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters('train')
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    do_save = (
        (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
        )
        or (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
    )
    do_validate = (
        (
            (not end_of_epoch and do_save)  # validate during mid-epoch saves
            or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        )
        and not args.disable_validation
    )

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    max_update = args.max_update or math.inf
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or trainer.get_num_updates() >= max_update
    )

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, 'tpu', False):
            itr = tpu_data_loader(args, itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        if not getattr(args, 'tpu', False):
            # fallback for single node with multiple GPUs
            assert args.distributed_world_size <= torch.cuda.device_count()
            port = random.randint(10000, 20000)
            args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
            args.distributed_rank = None  # set based on device id
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, ),
                nprocs=args.distributed_world_size,
            )
        else:
            import torch_xla.distributed.xla_multiprocessing as xmp
            torch.multiprocessing.set_sharing_strategy('file_system')
            xmp.spawn(
                fn=distributed_main,
                args=(args, ),
                nprocs=8,  # use all 8 TPU cores
            )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
