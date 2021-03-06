# Use tensorboard

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from datetime import datetime
import os
import torchvision.models as tmodels

#import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    try:
        if classname.find('Conv') != -1:
            init.xavier_uniform(m.weight)
        elif classname.find('Linear') != -1:
            init.xavier_uniform(m.weight)
    except:
        pass

def build_cnn(opt):
    net  = tmodels.resnet152(pretrained=True)
    net = nn.Sequential(\
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4)
    print('###freeze layers###')
    for name, param in net.named_parameters():
        if 'bn' in name:
           param.requires_grad = False
           print (name,end=' ')
    print ('')
    print ('###end of list###')
    try:
      if vars(opt).get('start_from', None) is not None:
        net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn.pth')))
    except:
        print('load save failed')
    return net


def train(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = infos.get('val_result_history', {})
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    ss_prob_history = infos.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    cnn_model = build_cnn(opt)
    cnn_model.cuda()
    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()
    best_val_score = 0

    crit = utils.LanguageModelCriterion()

    metric_history = []

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    for name, param in model.named_parameters():
        print(name)
    if opt.finetune_cnn_after != -1:
        # only finetune the layer2 to layer4
        trainable_param = []
        for module in cnn_model._modules.values()[5:]:
            trainable_param += filter(lambda p: p.requires_grad, module.parameters())
        print('number of trainable param',len(trainable_param))
        cnn_optimizer = optim.Adam(trainable_param, lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))
    #model.apply(weights_init)
    ttx = 0
    loss_record = 0.0
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = float(epoch - opt.learning_rate_decay_start) / float(opt.learning_rate_decay_every)
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = 1e-4 * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Update the training stage of cnn
            if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
                for p in cnn_model.parameters():
                    p.requires_grad = False
                cnn_model.eval()
            else:
                for p in cnn_model.parameters():
                    p.requires_grad = True
                # Fix the first few layers:
                for module in cnn_model._modules.values()[:5]:
                    for p in module.parameters():
                        p.requires_grad = False
                for name, param in cnn_model.named_parameters():
                    if 'bn' in name:
                        param.requires_grad = False
                cnn_model.train()
                #utils.set_lr(optimizer, 1e-4)
            update_lr_flag = False

        torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        data['images'] = utils.prepro_images(data['images'], True)
        torch.cuda.synchronize()
        #print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = cnn_model(images).permute(0, 2, 3, 1)
        fc_feats = att_feats.mean(1).mean(2).squeeze(1).squeeze(1)

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), opt.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * opt.seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), opt.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * opt.seq_per_img,) + fc_feats.size()[1:]))

        optimizer.zero_grad()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            cnn_optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            utils.clip_gradient(cnn_optimizer, opt.grad_clip)
            cnn_optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()

        loss_record += train_loss

        ttx += 1

        if ttx % 100 == 0:
            print(datetime.now()),
            print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                  .format(iteration, epoch, loss_record/100.0, end - start))
            loss_record = 0.0
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader, eval_kwargs)

            # Write validation result into summary
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss
            metric_history.append(lang_stats['CIDEr'])
            print('##########################')
            #print('now metric = ', lang_stats['CIDEr'])
            print('metric history:')
            for ii in range(len(metric_history)):
                print('epoch',ii, metric_history[ii])
            print('##########################')

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn.pth')
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['val_result_history'] = val_result_history
                infos['loss_history'] = loss_history
                infos['lr_history'] = lr_history
                infos['ss_prob_history'] = ss_prob_history
                infos['vocab'] = loader.get_vocab()
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path, 'model-cnn-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    print("cnn model saved to {}".format(cnn_checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
