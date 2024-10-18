import os
import numpy as np
import torch
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity
from torch.linalg import norm
torch.set_printoptions(precision=2)

class ContinuallyLearningPrototypes(nn.Module):
    """
    This is an implementation of the Nearest Class Mean algorithm for streaming learning.
    """

    def __init__(self,
                 input_shape,
                 backbone=None,
                 sim_metric='dot_product',
                 n_protos=500,
                 num_classes=2,
                 alpha_init=1,
                 sim_th_init=0.45,
                 n_wta=5,
                 k_hit=1,
                 k_miss=0.5,
                 tau_sim_th_pos=100,
                 tau_sim_th_neg=100,
                 k_sim_th_pos=0.9,
                 k_sim_th_neg=1.1,
                 device='cuda',
                 supervised=True,
                 adaptive_th=False,
                 adaptive_protos=True,
                 learn_outliers=True,
                 verbose=0):
        """
        Init function for the CLP model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(ContinuallyLearningPrototypes, self).__init__()

        # CLP parameters
        self.input_shape = input_shape
        self.sim_metric = sim_metric
        self.n_protos = n_protos
        self.num_classes = num_classes
        self.alpha_init = alpha_init
        self.sim_th_init = sim_th_init
        self.n_wta = n_wta
        self.k_hit = k_hit
        self.k_miss = k_miss
        self.tau_sim_th_pos = tau_sim_th_pos
        self.tau_sim_th_neg = tau_sim_th_neg
        self.k_sim_th_pos = k_sim_th_pos
        self.k_sim_th_neg = k_sim_th_neg
        self.device = device
        self.verbose = verbose
        self.adaptive_th = adaptive_th
        self.adaptive_protos = adaptive_protos
        self.supervised = supervised
        self.learn_outliers = learn_outliers

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights for CLP
        self.prototypes_ = torch.zeros(self.n_protos, input_shape).to(self.device)
        self.proto_labels_ = -1 * torch.ones((self.n_protos, 1)).long().to(self.device)
        self.alphas_ = self.alpha_init * torch.ones((self.n_protos, 1)).to(self.device)
        self.sim_th_ = self.sim_th_init * torch.ones((self.n_protos, 1)).to(self.device)
        self.goodness_ = torch.ones((self.n_protos, 1)).to(self.device)
        self.classes_ = []
        self.next_alloc_id_ = 0

        self.n_outlier = 0
        self.n_error = 0
        self.num_updates = 0


    @torch.no_grad()
    def fit(self, x, y, i=0):
        """
        Fit the NCM model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        x = x[None, :]

        self.mistaken_proto_inds_ = []  # inds of protos that made incorrect inferences for current sample
        n_mistakes = 0
        # y = torch.tensor(y).float()
        # x = torch.tensor(x).float()

        if self.sim_metric == 'dot_product':  # normalize x, if we use dot product similarity
            x = x / norm(x, 2)

        while True:

            inds_sorted = 0
            sims_sorted = 0
            similarities = self._calc_similarities(x)
            sims = similarities.clone().detach()

            th_passing_check = torch.gt(self.sim_th_, sims)
            sims[th_passing_check] = 0
            n_th_passing_protos = self.n_protos - torch.sum(th_passing_check)
            if torch.sum(sims) > 0:
                sims_sorted, inds_sorted = torch.sort(sims, 0, descending=True)
                bmu_ind = inds_sorted[0]
                bmu_sim = sims_sorted[0]
            else:
                bmu_ind = -1
                bmu_sim = 0
                sims_sorted = 0

            if (y.item() not in self.classes_) and self.supervised:
                self.classes_.append(y.item())
                if self.verbose >= 1:
                    print("Novel Label!")
                self._allocate(x, y)
                break

            # Novel instance --> Allocate
            # if no winner, because all similarities are below the given threshold, then allocate
            if bmu_ind == -1:
                if self.verbose >= 1:
                    print("Novel Instance!")
                    print("Label", y)
                self._allocate(x, y)
                break

            # Get the winner prototype
            bmu = self.prototypes_[[bmu_ind]]

            # Calculate Error
            error = self._calc_err(x, bmu)
            

            # TODO: Implement for the cases where pseudo-labeling is enabled
            # If winner not assigned to a label, then assign it to
            # the training instance's label
            # if self.proto_labels_[[bmu_ind]] == -1:
            #     if self.verbose >= 1:
            #         print("Unsupervised allocating...")
            #     self.proto_labels_[[bmu_ind]] = y
            #     # self.prototypes_[[bmu_ind]] += self.alphas_[[bmu_ind]] * error
            #     # self.prototypes_[[bmu_ind]] = self.prototypes_[[bmu_ind]]/torch.norm(self.prototypes_[[bmu_ind]], p=2)
            #
            #     # update the threshold towards max_sim-eps
            #     # self.sim_th_[[bmu_ind]] = self.sim_th_[[bmu_ind]] + \
            #     # (self.k_sim_th_pos*max_sim - self.sim_th_[[bmu_ind]]) / self.tau_sim_th_pos
            #
            #     self.hits_[[bmu_ind]] += 1
            #     # self.alphas_[[bmu_ind]] = self.alpha_init/self.hits_[[bmu_ind]]
            #     break

            # Update the winner based on its inference
            # If CORRECT prediction
            if self.supervised:
                if self.proto_labels_[[bmu_ind]] == y:
                    if self.adaptive_protos:
                        self._positive_update(bmu_ind, bmu_sim, error)
                    break

                # if INCORRECT prediction
                else:
                    self.n_error += 1
                    if self.adaptive_protos:
                        n_protos_to_update = min(self.n_wta, n_th_passing_protos)
                        positive_match = False
                        for m in range(0,n_protos_to_update):
                            next_bmu_ind = inds_sorted[m]
                            next_sim = sims_sorted[m]
                            error = self._calc_err(x, self.prototypes_[[next_bmu_ind]])
                            if self.proto_labels_[[next_bmu_ind]] == y:
                                self._positive_update(bmu_ind, next_sim, error)
                                positive_match = True
                            else:
                                self._negative_update(bmu_ind, next_sim, error)
                                if m==n_protos_to_update-1 and not positive_match:
                                    # Allocate a new prototype if none of k-winners is positive match
                                    self.n_outlier += 1
                                    if self.learn_outliers:
                                        self._allocate(x, y)
                    else:
                        self._allocate(x, y)

                    break

            else:
                if self.adaptive_protos:
                    self._positive_update(bmu_ind, bmu_sim, error)
                break

                # if self.verbose >= 1:
                #     print("Predicted:", self.proto_labels_[[bmu_ind]].item(), " Actual: ", y)
                #     print("sim_th & alpha:", self.sim_th_[[bmu_ind]].item(), self.alphas_[[bmu_ind]].item())

                # TODO: implement forgetting
                # # If more misses than hits, then forget this prototype, reset it
                # if self.alphas_[[bmu_ind]] > 1:
                #     self._forget(bmu_ind)
            
        self.num_updates += 1

    
    def _positive_update(self, bmu_ind, sim, error):
        self.prototypes_[[bmu_ind]] += self.alphas_[[bmu_ind]] * error
        self.prototypes_[[bmu_ind]] = self.prototypes_[[bmu_ind]] / torch.norm(self.prototypes_[[bmu_ind]], p=2)

        # update the threshold towards max_sim-eps
        if self.adaptive_th:
            self.sim_th_[[bmu_ind]] = self.sim_th_[[bmu_ind]] + \
            (self.k_sim_th_pos*sim - self.sim_th_[[bmu_ind]]) / self.tau_sim_th_pos

        if self.supervised:
            self.goodness_[[bmu_ind]] += self.k_hit
        else:
            self.goodness_[[bmu_ind]] += 0.5*self.k_hit

        self.alphas_[[bmu_ind]] = self.alpha_init / max(self.goodness_[[bmu_ind]],1)

    def _negative_update(self, bmu_ind, sim, error):
        # update the mistaken prototype
        self.prototypes_[[bmu_ind]] -= self.alphas_[[bmu_ind]] * error
        self.prototypes_[[bmu_ind]] = self.prototypes_[[bmu_ind]] / torch.norm(self.prototypes_[[bmu_ind]], p=2)

        # update the threshold of this prototype
        if self.adaptive_th:
            self.sim_th_[[bmu_ind]] = self.sim_th_[[bmu_ind]] + \
            (self.k_sim_th_neg*sim - self.sim_th_[[bmu_ind]]) / self.tau_sim_th_neg

        self.goodness_[[bmu_ind]] -= self.k_miss
        self.alphas_[[bmu_ind]] = self.alpha_init / max(self.goodness_[[bmu_ind]],1)

    @torch.no_grad()
    def predict(self, X, return_probas=False, thresholded=False, return_sims=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :param thresholded: True if similarity thresholds are enforced.
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        if self.sim_metric == 'dot_product':  # normalize x, if we use dot product similarity
            X = X / norm(X, dim=1).unsqueeze(1)
            
        sims = self._calc_similarities(X).detach()
        
        if thresholded:
            th_passing_check = torch.gt(self.sim_th_.tile((1, sims.shape[1])), sims)
            sims[th_passing_check] = 0

        # Create a mask for each unique label
        label_masks = [self.proto_labels_ == label for label in range(self.n_protos)]

        # Calculate the maximum score for each class using masking and max
        scores = torch.zeros(size=(X.shape[0],self.num_classes))

        # learned classes
        learned_classes = torch.unique(self.proto_labels_[self.proto_labels_ > -1])

        # Apply the mask and take the maximum for each class
        label_masks = torch.squeeze(torch.stack(label_masks))
        for c in learned_classes:
            class_mask = label_masks[c, :]
            class_scores = sims[class_mask, :]
            scores[:, c] = class_scores.max(dim=0)[0]
            
        # return predictions or probabilities
        if return_sims:
            return sims[:self.next_alloc_id_,:].T
        elif not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()
        
    def _calc_err(self, x, bmu):
        error = 0
        if self.sim_metric == 'euclidean':
            error = x - bmu

        elif self.sim_metric == 'dot_product':
            error = x
        else:
            raise NotImplementedError("Can't compute error for cosine similarity: only implemented for Euclidean and dot product similarity")
            
        return error
    
    def _calc_similarities(self, x):

        similarities = 0
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # x = x.float()

        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        if self.sim_metric == 'euclidean':
            similarities = -torch.cdist(self.prototypes_, x, p=2)

        elif self.sim_metric == 'dot_product':
            similarities = torch.mm(self.prototypes_, x.T)

        elif self.sim_metric == 'cosine':
            similarities = pairwise_cosine_similarity(self.prototypes_, x)
        return similarities

    def _allocate(self, x, y):
        # print("Mistake again, allocating...")
        bmu_ind = self.next_alloc_id_
        self.proto_labels_[[bmu_ind]] = y

        error = x - self.prototypes_[[bmu_ind]]
        self.prototypes_[[bmu_ind]] += self.alphas_[[bmu_ind]] * error
        self.prototypes_[[bmu_ind]] = self.prototypes_[[bmu_ind]] / torch.norm(self.prototypes_[[bmu_ind]], p=2)

        self.goodness_[[bmu_ind]] += 1
        # self.alphas_[[bmu_ind]] = self.alpha_init / self.hits_[[bmu_ind]]
        self.next_alloc_id_ = min(self.next_alloc_id_ + 1, self.n_protos - 1)
        # print("Total number of allocated prototypes:", self.next_alloc_id_)

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, return_probas=False, thresholded=False)

    @torch.no_grad()
    def evaluate_ood_(self, test_loader):
        print('\nTesting OOD on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        scores = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            ood_scores = self.ood_predict(batch_x_feat)
            end = start + ood_scores.shape[0]
            scores[start:end] = ood_scores
            labels[start:end] = test_y.squeeze()
            start = end

        return scores, labels

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit NCM one example at a time

        for x, y in zip(batch_x, batch_y):
            self.fit(x[None, :], y)

    @torch.no_grad()
    def train_(self, train_loader):
        # print('\nTraining on %d images.' % len(train_loader.dataset))

        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            self.fit_batch(batch_x_feat, batch_y, batch_ix)

    @torch.no_grad()
    def evaluate_(self, test_loader,return_probas=True, thresholded=False, return_sims=False):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)

        if return_sims:
            probabilities = torch.empty((num_samples, self.next_alloc_id_))
        else: 
            probabilities = torch.empty((num_samples, self.num_classes))

        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, 
                                  return_probas=return_probas, 
                                  thresholded=thresholded,
                                  return_sims=return_sims)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = test_y.squeeze()
            start = end
        return probabilities, labels

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()

        d['prototypes_'] = self.prototypes_.cpu()
        d['proto_labels_'] = self.proto_labels_.cpu()
        d['alphas_'] = self.alphas_.cpu()
        d['sim_th_'] = self.sim_th_.cpu()
        d['goodness_'] = self.goodness_.cpu()
        d['classes_'] = self.classes_
        d['next_alloc_id_'] = self.next_alloc_id_

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        print('\nloading ckpt from: %s' % save_file)
        d = torch.load(os.path.join(save_file))
        self.prototypes_ = d['prototypes_'].to(self.device)
        self.proto_labels_ = d['proto_labels_'].to(self.device)
        self.alphas_ = d['alphas_'].to(self.device)
        self.sim_th_ = d['sim_th_'].to(self.device)
        self.goodness_ = d['goodness_'].to(self.device)
        self.classes_ = d['classes_']
        self.next_alloc_id_ = d['next_alloc_id_']

    # Locate the best matching unit
    def _get_best_matching_unit(self, x):

        similarities = self._calc_similarities(x)
        similarities[self.mistaken_proto_inds_] -= 10000
        sims = similarities.clone().detach()

        th_passing_check = torch.gt(sims, self.sim_th_.tile((1, sims.shape[1])))
        sims_sorted, inds_sorted = torch.sort(sims, 0, descending=True)
        th_passing_sorted = torch.gather(th_passing_check, 0, inds_sorted)

        bmu_inds = torch.zeros(size=(1, sims.shape[1]))
        max_sims = torch.zeros(size=(1, sims.shape[1]))

        for i in range(sims.shape[1]):
            top_th_passing_inds = inds_sorted[th_passing_sorted[:, i], i]
            max_th_passing_sims = sims_sorted[th_passing_sorted[:, i], i]
            if len(top_th_passing_inds) > 0:
                bmu_inds[0, i] = top_th_passing_inds[0]
                max_sims[0, i] = max_th_passing_sims[0]
            else:
                bmu_inds[0, i] = -1
                max_sims[0, i] = 0

        inds_sorted = inds_sorted.long()
        top_sims, top_inds = sims_sorted[:5, :], inds_sorted[:5, :]
        if self.verbose == 2:
            for i in range(0, top_sims.shape[1], 3):
                print("-----------------------------------------------------------")
                print("sims:  ", top_sims[:, i].t().data)
                print("simth: ", self.sim_th_[top_inds[:, i]].t().data)
                print("labels:", self.proto_labels_[top_inds[:, i]].t().data)
                print("alphas:", self.alphas_[top_inds[:, i]].t().data)

        bmu_inds = bmu_inds.squeeze()
        max_sims = max_sims.squeeze()

        return bmu_inds.long(), max_sims