"""
多头注意力版本的 STAGATE 模型
"""
import tensorflow.compat.v1 as tf

class GATE:
    def __init__(self, hidden_dims, num_heads=4, alpha=0, nonlinear=True, weight_decay=0.0001):
        """
        hidden_dims: e.g., [input_dim, 512, 30]
        num_heads: number of attention heads
        alpha: mix weight for prune graph (kept for compatibility)
        nonlinear: apply ELU between layers except last
        weight_decay: L2 regularization
        """
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.num_heads = num_heads
        self.W, self.v, self.prune_v = self.define_weights(hidden_dims, num_heads)
        self.C = {}
        self.prune_C = {}
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay

    def __call__(self, A, prune_A, X, corr_target=None):
        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self.__encoder(A, prune_A, H, layer)
            if self.nonlinear and layer != self.n_layers - 1:
                H = tf.nn.elu(H)
        self.H = H

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
            if self.nonlinear and layer != 0:
                H = tf.nn.elu(H)
        X_ = H

        # Feature reconstruction loss
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        weight_decay_loss = 0
        for layer in range(self.n_layers):
            weight_decay_loss += tf.multiply(tf.nn.l2_loss(self.W[layer]), self.weight_decay, name='weight_loss')

        if corr_target is not None:
            att_values_all_heads = []
            for head in range(self.num_heads):
                att_values_all_heads.append(self.C[0][head].values)
            att_values = tf.reduce_mean(tf.stack(att_values_all_heads, axis=0), axis=0)

            mask = tf.not_equal(corr_target, 0.0)
            masked_att = tf.boolean_mask(att_values, mask)
            masked_target = tf.boolean_mask(corr_target, mask)

            def _corr_on_masked():
                return self._pearson_corr(masked_att, masked_target)

            corr = tf.cond(tf.size(masked_target) > 0, _corr_on_masked, lambda: tf.constant(0.0, dtype=tf.float32))
            self.loss = (1.0 - corr) + weight_decay_loss
        else:
            self.loss = features_loss + weight_decay_loss

        if self.alpha == 0:
            self.Att_l = self.C
        else:
            self.Att_l = {'C': self.C, 'prune_C': self.prune_C}
        return self.loss, self.H, self.Att_l, X_

    def __encoder(self, A, prune_A, H, layer):
        H = tf.matmul(H, self.W[layer])
        if layer == self.n_layers - 1:
            return H

        self.C[layer] = self.multi_head_graph_attention_layer(A, H, self.v[layer], layer)

        if self.alpha == 0:
            H_heads = [tf.sparse_tensor_dense_matmul(self.C[layer][head], H) for head in range(self.num_heads)]
            return tf.reduce_mean(tf.stack(H_heads, axis=0), axis=0)
        else:
            self.prune_C[layer] = self.multi_head_graph_attention_layer(prune_A, H, self.prune_v[layer], layer)
            H_main = [tf.sparse_tensor_dense_matmul(self.C[layer][head], H) for head in range(self.num_heads)]
            H_prune = [tf.sparse_tensor_dense_matmul(self.prune_C[layer][head], H) for head in range(self.num_heads)]
            H_main_avg = tf.reduce_mean(tf.stack(H_main, axis=0), axis=0)
            H_prune_avg = tf.reduce_mean(tf.stack(H_prune, axis=0), axis=0)
            return (1 - self.alpha) * H_main_avg + self.alpha * H_prune_avg

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        if layer == 0:
            return H

        if self.alpha == 0:
            H_heads = [tf.sparse_tensor_dense_matmul(self.C[layer - 1][head], H) for head in range(self.num_heads)]
            return tf.reduce_mean(tf.stack(H_heads, axis=0), axis=0)
        else:
            H_main = [tf.sparse_tensor_dense_matmul(self.C[layer - 1][head], H) for head in range(self.num_heads)]
            H_prune = [tf.sparse_tensor_dense_matmul(self.prune_C[layer - 1][head], H) for head in range(self.num_heads)]
            H_main_avg = tf.reduce_mean(tf.stack(H_main, axis=0), axis=0)
            H_prune_avg = tf.reduce_mean(tf.stack(H_prune, axis=0), axis=0)
            return (1 - self.alpha) * H_main_avg + self.alpha * H_prune_avg

    def define_weights(self, hidden_dims, num_heads):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i + 1]))

        Ws_att = {}
        for i in range(self.n_layers - 1):
            Ws_att[i] = []
            for head in range(num_heads):
                v = {}
                v[0] = tf.get_variable("v%s_head%s_0" % (i, head), shape=(hidden_dims[i + 1], 1))
                v[1] = tf.get_variable("v%s_head%s_1" % (i, head), shape=(hidden_dims[i + 1], 1))
                Ws_att[i].append(v)

        if self.alpha == 0:
            return W, Ws_att, None

        prune_Ws_att = {}
        for i in range(self.n_layers - 1):
            prune_Ws_att[i] = []
            for head in range(num_heads):
                prune_v = {}
                prune_v[0] = tf.get_variable("prune_v%s_head%s_0" % (i, head), shape=(hidden_dims[i + 1], 1))
                prune_v[1] = tf.get_variable("prune_v%s_head%s_1" % (i, head), shape=(hidden_dims[i + 1], 1))
                prune_Ws_att[i].append(prune_v)

        return W, Ws_att, prune_Ws_att

    def _pearson_corr(self, a, b, eps=1e-8):
        a = tf.reshape(a, [-1])
        b = tf.reshape(b, [-1])
        a_center = a - tf.reduce_mean(a)
        b_center = b - tf.reduce_mean(b)
        num = tf.reduce_sum(a_center * b_center)
        den = tf.sqrt(tf.reduce_sum(tf.square(a_center)) * tf.reduce_sum(tf.square(b_center)) + eps)
        return num / den

    def multi_head_graph_attention_layer(self, A, M, v_list, layer):
        attentions = []
        for head in range(self.num_heads):
            with tf.variable_scope("layer_%s_head_%s" % (layer, head)):
                v = v_list[head]
                f1 = tf.matmul(M, v[0])
                f1 = A * f1
                f2 = tf.matmul(M, v[1])
                f2 = A * tf.transpose(f2, [1, 0])
                logits = tf.sparse_add(f1, f2)

                unnormalized_attentions = tf.SparseTensor(
                    indices=logits.indices,
                    values=tf.nn.sigmoid(logits.values),
                    dense_shape=logits.dense_shape,
                )
                att = tf.sparse_softmax(unnormalized_attentions)
                att = tf.SparseTensor(indices=att.indices, values=att.values, dense_shape=att.dense_shape)
                attentions.append(att)
        return attentions

    def graph_attention_layer(self, A, M, v, layer):
        # Compatibility shim: single-head interface
        return self.multi_head_graph_attention_layer(A, M, [v], layer)[0]
