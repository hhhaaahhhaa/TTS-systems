import numpy as np
from tqdm import tqdm
from typing import List
import time

from sklearn.cluster import KMeans

from dlhlp_lib.utils import batchify, torch_exist_nan
from dlhlp_lib.s3prl import S3PRLExtractor


# Simple implementation of dynamic programming based phoneme segmentation method given in
#   Towards unsupervised phone and word segmentation using self-supervised vector-quantized neural networks
#   (https://arxiv.org/abs/2012.07551, INTERSPEECH 2021)
# Author: Yuan Tseng (https://github.com/roger-tseng)
def segment(reps, distance_array, pen, lambd=35):
    alphas = [[0, None]]

    # Perform dynamic-programming-based segmentation
    for t in range(1,reps.shape[0]+1):

        errors = []
        closest_centers = []
        
        for segment_length in range(1,t+1):

            # array len = num of clusters
            # ith element is sum of distance from the last segment_length tokens until Tth token to the ith cluster center
            distance_subarray = distance_array[t-segment_length:t].sum(axis=0)

            closest_center = distance_subarray.argmin()
            error = alphas[t-segment_length][0] + distance_subarray.min() + lambd * pen(segment_length)

            closest_centers.append(closest_center)
            errors.append(error)

        errors = np.array(errors)
        alpha, a_min, closest = errors.min(), t-1-errors.argmin(), closest_centers[errors.argmin()]
        alphas.append([alpha, a_min, closest])

    # Backtrack to find optimal boundary tokens and label
    boundaries = []
    label_tokens = []
    tk = len(alphas)-1
    while (tk!=0):
        boundaries.append(tk)
        label_tokens.append(alphas[tk][2])
        tk = alphas[tk][1]  
    boundaries.reverse()
    label_tokens.reverse()

    if lambd == 0:  # merge repeat tokens (when lambd=0 cosecutive tokens may repeat)
        cur_token = None
        new_boundaries, new_label_tokens = [], []
        for i in range(len(boundaries)):
            if i == 0:
                cur_token = label_tokens[0]
            if i == len(boundaries) - 1:
                new_boundaries.append(boundaries[i])
                new_label_tokens.append(cur_token)
                return new_boundaries, new_label_tokens
            if label_tokens[i + 1] != cur_token:
                new_boundaries.append(boundaries[i])
                new_label_tokens.append(cur_token)
                cur_token = label_tokens[i + 1]
            
    return boundaries, label_tokens


def segment_by_kmeans_model(reps, kmeans_model, pen, lambd=35):
    '''
    Inputs:
    reps        :   Representation sequence from self supervised model
    kmeans_model:   Pretrained scikit-learn MiniBatchKMeans model
    pen         :   penalty function penalizing segment length (longer segment, higher penalty)
    lambd       :   penalty weight (larger weight, longer segment)

    Outputs:
    boundaries  :   List of tokens at right boundaries of segments 
                    (assuming token sequence starts from 1 to Tth token)
    label_token :   List of token labels for segments

    e.g. :

    If  tokens = [34, 55, 62, 83, 42]
        boundaries = [3, 5]
        label_token = [55, 83]

    then segmentation is :
    | 34 55 62 | 83 42 |
    |    55    |   83  |

    '''
    
    # array of distances to closest cluster center, size: token sequence len * num of clusters
    distance_array = np.square( kmeans_model.transform(reps) )
    return segment(reps, distance_array, pen, lambd=lambd)


def calculate_ssl_centroids(extractor: S3PRLExtractor, n_clusters: int, wav_paths: List[str], layer: int, batch_size=16, norm=False) -> KMeans:
    all_frames = []
    gen = batchify(wav_paths, batch_size)
    for batch_paths in tqdm(gen):
        reprs, n_frames = extractor.extract_from_paths(batch_paths, norm)
        for wav_path, repr, n_frame in zip(batch_paths, reprs, n_frames):
            sliced_repr = repr[:n_frame, layer, :].clone()  # L, dim
            all_frames.append(sliced_repr.detach().cpu().numpy())

    # Concatenate and perform KMeans clustering.
    st = time.time()
    print("Perform KMeans...")
    all_frames = np.concatenate(all_frames, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_frames)
    print(f"Done in {time.time()-st:.2f}s. Data {all_frames.shape} => Centroids {kmeans.cluster_centers_.shape}.")

    return kmeans


def calculate_ssl_postnet_centroids(extractor: S3PRLExtractor, n_clusters: int, wav_paths: List[str], postnet, batch_size=16, norm=False) -> KMeans:
    all_frames = []
    gen = batchify(wav_paths, batch_size)
    for batch_paths in tqdm(gen):
        reprs, n_frames = extractor.extract_from_paths(batch_paths, norm)
        reprs = postnet(reprs)
        for wav_path, repr, n_frame in zip(batch_paths, reprs, n_frames):
            sliced_repr = repr[:n_frame, :].clone()  # L, dim
            all_frames.append(sliced_repr.detach().cpu().numpy())

    # Concatenate and perform KMeans clustering.
    st = time.time()
    print("Perform KMeans...")
    all_frames = np.concatenate(all_frames, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_frames)
    print(f"Done in {time.time()-st:.2f}s. Data {all_frames.shape} => Centroids {kmeans.cluster_centers_.shape}.")

    return kmeans


class DPDPSSLUnit(object):
    """
    DPDP algorithm for SSL unit generation using s3prl toolkit, support adding postnet module.
    If postnet module is specified, then layer argument will be ignored.
    """

    @staticmethod
    def default_pen(segment_length):
        return 1 - segment_length

    def __init__(self, s3prl_name, layer=0, norm=False, postnet=None):
        self.debug = False
        self._extractor = S3PRLExtractor(s3prl_name)
        self._layer = layer
        self._norm = norm
        self._pen = DPDPSSLUnit.default_pen

        self._postnet = lambda x: x[:, :, layer]
        if postnet is not None:
            self._postnet = postnet
    
    def cuda(self):
        self._extractor.cuda()
    
    def cpu(self):
        self._extractor.cpu()

    def set_penalize_function(self, func):
        self._pen = func
    
    def calculate_ssl_centroids(self, n_clusters, wav_paths, batch_size=16) -> KMeans:
        kmeans_model = calculate_ssl_postnet_centroids(self._extractor, n_clusters, wav_paths, self._postnet, batch_size, self._norm)
        return kmeans_model

    def segment(self, wav_path: str, kmeans_model: KMeans, lambd=35):
        repr, n_frame = self._extractor.extract_from_paths([wav_path], norm=self._norm)        
        repr = self._postnet(repr)
        sliced_repr = repr[0].detach().cpu()  # L, dim
        try:
            assert not torch_exist_nan(sliced_repr)
        except:
            self.log("NaN in SSL feature:")
            self.log(wav_path)
            raise ValueError
        
        sliced_repr = sliced_repr.numpy()
        if self.debug:
            tokens = kmeans_model.predict(sliced_repr).tolist()
            self.log(f"tokens: {tokens}")
            self.log(f"len(tokens): {len(tokens)}")

        boundaries, label_tokens = segment_by_kmeans_model(sliced_repr, kmeans_model, self._pen, lambd=lambd)
        if self.debug:
            self.log(f"boundaries: {boundaries}")
            self.log(f"label_tokens: {label_tokens}")
            self.log(f"Num of segments = {len(label_tokens)}")
            print()

        foramtted_boundaries = []
        st = 0.0
        for b in boundaries:
            foramtted_boundaries.append((st, b * self._extractor._fp / 1000))
            st = b * self._extractor._fp / 1000
        
        return foramtted_boundaries, label_tokens

    def segment_by_dist(self, wav_path: str, lambd=35):
        repr, n_frame = self._extractor.extract_from_paths([wav_path], norm=self._norm) 
        repr = self._postnet(repr)
        sliced_repr = repr[0].detach().cpu()  # L, dim
        try:
            assert not torch_exist_nan(sliced_repr)
        except:
            self.log("NaN in SSL feature:")
            self.log(wav_path)
            raise ValueError
        
        sliced_repr = sliced_repr.numpy()
        if self.debug:
            tokens = sliced_repr.argmin(axis=1).tolist()
            self.log(f"tokens: {tokens}")
            self.log(f"len(tokens): {len(tokens)}")

        boundaries, label_tokens = segment(sliced_repr, sliced_repr, self._pen, lambd=lambd)
        if self.debug:
            self.log(f"boundaries: {boundaries}")
            self.log(f"label_tokens: {label_tokens}")
            self.log(f"Num of segments = {len(label_tokens)}")
            print()

        foramtted_boundaries = []
        st = 0.0
        for b in boundaries:
            foramtted_boundaries.append((st, b * self._extractor._fp / 1000))
            st = b * self._extractor._fp / 1000
        
        return foramtted_boundaries, label_tokens
    
    def batch_segment(self, wav_paths: List[str], kmeans_model: KMeans, batch_size=16, lambd=35):
        """
        Performing segment() batchwise.
        """
        foramtted_boundaries_list, label_tokens_list = [], []
        for batch_paths in batchify(wav_paths, batch_size):
            reprs, n_frames = self._extractor.extract_from_paths(batch_paths, norm=self._norm)
            reprs = self._postnet(reprs)
            for wav_path, repr, n_frame in zip(batch_paths, reprs, n_frames):
                sliced_repr = repr[:n_frame, :].clone()  # L, dim
                try:
                    assert not torch_exist_nan(sliced_repr)
                except:
                    self.log("NaN in SSL feature:")
                    self.log(wav_path)
                    continue
                sliced_repr = sliced_repr.detach().cpu().numpy()
                tokens = kmeans_model.predict(sliced_repr).tolist()
                if self.debug:
                    self.log(f"tokens: {tokens}")
                    self.log(f"len(tokens): {len(tokens)}")

                boundaries, label_tokens = segment_by_kmeans_model(sliced_repr, kmeans_model, self._pen, lambd=lambd)
                if self.debug:
                    self.log(f"boundaries: {boundaries}")
                    self.log(f"label_tokens: {label_tokens}")
                    self.log(f"Num of segments = {len(label_tokens)}")

                foramtted_boundaries = []
                st = 0.0
                for b in boundaries:
                    foramtted_boundaries.append((st, b * self._extractor._fp / 1000))
                    st = b * self._extractor._fp / 1000
                
                foramtted_boundaries_list.append(foramtted_boundaries)
                label_tokens_list.append(label_tokens)
        
        return foramtted_boundaries_list, label_tokens_list
    
    def log(self, msg):
        print(f"[DPDP]: ", msg)
