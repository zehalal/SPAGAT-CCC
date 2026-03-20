import tensorflow.compat.v1 as tf

# Force TF2 to run in TF1 graph mode (placeholders/sessions).
tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
from .model_multihead import GATE
from tqdm import tqdm

class STAGATE:
    def __init__(self, hidden_dims, num_heads=4, alpha=0, n_epochs=500, lr=0.0001,
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001,
                 verbose=True, random_seed=2021, use_corr_loss=False):
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        self.loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.alpha = alpha
        self.use_corr_loss = use_corr_loss
        self.num_heads = num_heads
        self.gate = GATE(hidden_dims, num_heads, alpha, nonlinear, weight_decay)
        corr_arg = self.corr_target if self.use_corr_loss else None
        self.loss, self.H, self.C, self.ReX = self.gate(self.A, self.prune_A, self.X, corr_arg)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.prune_A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)
        self.corr_target = tf.placeholder(dtype=tf.float32, shape=[None], name="corr_target")

    def build_session(self, gpu=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu is False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, prune_A, X, corr_target=None):
        for epoch in tqdm(range(self.n_epochs)):
            self.run_epoch(epoch, A, prune_A, X, corr_target)

    def run_epoch(self, epoch, A, prune_A, X, corr_target=None):
        feed = {self.A: A, self.prune_A: prune_A, self.X: X}
        if corr_target is not None:
            feed[self.corr_target] = corr_target
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed)
        self.loss_list.append(loss)
        return loss

    def infer(self, A, prune_A, X, corr_target=None):
        feed = {self.A: A, self.prune_A: prune_A, self.X: X}
        if corr_target is not None:
            feed[self.corr_target] = corr_target
        H, C, ReX = self.session.run([self.H, self.C, self.ReX], feed_dict=feed)
        return H, self.combine_attention(C), self.loss_list, ReX

    def combine_attention(self, input_att):
        if self.alpha == 0:
            # 多头：每层是 list[head] 的 SparseTensor，需要返回 list[scipy sparse]
            combined = []
            for layer in input_att:
                # 对多头的注意力取平均
                head_mats = [sp.coo_matrix((head.values, (head.indices[:, 0], head.indices[:, 1])), shape=head.dense_shape)
                             for head in input_att[layer]]
                avg = sum(head_mats) / len(head_mats)
                combined.append(avg)
            return combined
        else:
            Att_C = []
            Att_pruneC = []
            for layer in input_att['C']:
                head_mats = [sp.coo_matrix((head.values, (head.indices[:, 0], head.indices[:, 1])), shape=head.dense_shape)
                             for head in input_att['C'][layer]]
                avg = sum(head_mats) / len(head_mats)
                Att_C.append(avg)
            for layer in input_att['prune_C']:
                head_mats = [sp.coo_matrix((head.values, (head.indices[:, 0], head.indices[:, 1])), shape=head.dense_shape)
                             for head in input_att['prune_C'][layer]]
                avg = sum(head_mats) / len(head_mats)
                Att_pruneC.append(avg)
            return [self.alpha * Att_pruneC[layer] + (1 - self.alpha) * Att_C[layer] for layer in range(len(Att_C))]
