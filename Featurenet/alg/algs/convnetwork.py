
import torch
import torch.nn.functional as F
from alg.modelopera import get_fea
from Featurenet.network import  common_network
from alg.algs.base import Algorithm


class FeatureNet(Algorithm):

    def __init__(self, args):

        super(FeatureNet, self).__init__(args)
        self.featurizer = get_fea(args)
        self.dprojection = common_network.feat_projection(
            self.featurizer.in_features, args.projection, args.layer)


        self.projection = common_network.feat_projection(
            self.featurizer.in_features, args.projection, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.projection, args.classifier)

        self.aprojection = common_network.feat_projection(
            self.featurizer.in_features, args.projection, args.layer)

        self.aclassifier = common_network.feat_classifier(
            args.num_classes*2, args.projection, args.classifier)
        if 'diff' in args:
            self.aclassifier = common_network.feat_classifier(
            args.num_classes * args.diff, args.projection, args.classifier)
        self.dclassifier = common_network.feat_classifier(
           2, args.projection, args.classifier)

        self.args = args


    def update_os(self, all_x, all_y, all_d, opt): 
        all_x1 = all_x.cuda().float()
        all_c1 = all_d.long()
        z1 = self.dprojection(self.featurizer(all_x1)) 
        cd1 = self.dclassifier(z1)
        ent_loss = F.cross_entropy(cd1, all_c1)  
        loss = ent_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis':0, 'ent': ent_loss.item()}




    def update_cs(self, all_x,all_y, opt,drop_log = 0.5):
        all_x = all_x.cuda().float()
        all_y = all_y.cuda().long()
        all_z = self.projection(self.featurizer(all_x))
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y) 
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        all_preds_prob = F.softmax(all_preds, dim=1)
        index_list = None

        # max_entropy, _ = torch.max(all_preds_prob , dim=1)

        # real_entropy = all_preds_prob [torch.arange(all_preds_prob .size(0)), all_y]
        # diff = max_entropy - real_entropy

        # indices = torch.nonzero(diff >  drop_log).squeeze()
        # index_list = indices.tolist()
        # if isinstance(index_list, list):
        #     pass
        # else:
        #     index_list = [indices.cpu().item()]

        return {'total': loss.item(), 'class': classifier_loss.item()},all_preds, index_list, all_z



    def update_ft(self,all_x, all_y, all_d, opt):
        all_x = all_x.cuda().float()
        all_c = all_y.long()
        all_d = all_d.long()
        all_y = all_d*self.args.num_classes+all_c  
        all_z = self.aprojection(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y) 
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}



    
    def predict(self, x):
        target_z = self.featurizer(x)
        return self.classifier(self.projection(target_z)),target_z

    def predict1(self, x):
        target_z = self.featurizer(x)
        return self.ddiscriminator(self.dprojection(target_z)),target_z
    def predict2(self, x):
        target_z = self.featurizer(x)
        return self.aclassifier(self.aprojection(target_z)),target_z
