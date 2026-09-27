# -*- coding: utf-8 -*-
"""AMAF agent for the MARL-UDS (Zhang et al. 2023, Water Research
229:119498) harness.

This is a port of the AMAF-DQN architecture (Kim et al.; original at
https://github.com/soon2soon/AMAF, src/amaf/networks/discrete.py,
class AMAFDQN) into the MARL-UDS agent interface so that it can be
trained and evaluated with the exact same environment, event sampling,
reward, and evaluation protocol as the paper's DQN/VDN/IQL baselines.

Architecture (matching the paper-frozen AMAF design):
  - shared feature trunk  : Dense(observ -> net_dim) + ReLU
  - value stream          : Dense(net_dim -> net_dim) + ReLU + Dense(-> 1)
  - H advantage heads     : each Dense(net_dim -> net_dim) + ReLU
                            + Dense(-> action_shape)   [H = n_heads]
  - gate (router)         : softmax over H heads conditioned on the
                            shared trunk (mode 'adaptive'), or uniform
                            1/H weights (mode 'uniform')
  - fusion                : 'uniform_anchored' anchors the fused
                            advantage to the uniform head-average with
                            a state-dependent trust tau(s) in (0,1)
                            initialized at trust_init; 'direct' fuses
                            purely by router weights
  - dueling combination   : Q = V + A_fused - mean(A_fused)

The class mirrors agent.dqn.DQN (Zhang) method-for-method: act(),
convert_action_to_setting(), update_net(), evaluate_net(),
episode_update(), save(), load(), plus the internal training steps, so
train_astlingen.py / test_astlingen.py drive it without any other
change than a config entry.

License: GPL-3.0 (inherited from MARL-UDS).
"""
from numpy import argmax, array
from os.path import join
import random

from tensorflow import (convert_to_tensor, expand_dims, float32,
                        GradientTape, one_hot, reduce_max, reduce_sum,
                        squeeze)
from tensorflow import keras as ks

from .qagent import QAgent


class AMAFAgent(QAgent):
    """Multi-head AMAF network wrapped in the QAgent interface.

    Follows QAgent.build_q_network() of Zhang's harness but extends
    the dueling advantage stream to H heads fused by a state-dependent
    gate. The model OUTPUT is identical in shape to Zhang's dueling
    DQN (action_shape+1 raw logits: 1 value + action_shape advantage
    post-Lambda), so act()/training code is unchanged.
    """

    def __init__(self, action_shape, observ_size, args, seq_len=None,
                 graph_conv=False):
        # Reuse QAgent for target bookkeeping only; we build our own
        # networks below (super().__init__ would build plain stacks).
        self.action_shape = action_shape
        self.observ_size = observ_size
        self.recurrent = True if seq_len is not None else False
        self.seq_len = seq_len

        self.net_dim = getattr(args, "net_dim", 128)
        self.num_layer = getattr(args, "num_layer", 3)
        self.hidden_dim = getattr(args, "hidden_dim", self.net_dim)
        self.dueling = getattr(args, "if_dueling", True)
        self.n_heads = int(getattr(args, "n_heads", 4))
        self.gate_mode = getattr(args, "gate_mode", "adaptive")
        self.fusion_mode = getattr(args, "fusion_mode", "uniform_anchored")
        self.trust_init = float(getattr(args, "trust_init", 0.1))

        if graph_conv:
            raise NotImplementedError(
                "AMAF port: graph convolution trunk not needed for the "
                "Astlingen 22-dim flat state used by the shipped DQN ckpt."
            )
        self.conv_layer = None
        self.graph_conv = False

        self.update_interval = getattr(args, "update_interval", 0.005)
        self.target_model = self.build_q_network()
        self.model = self.build_q_network()
        self.target_model.set_weights(self.model.get_weights())
        self.model_dir = args.cwd

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _trunk(self, x):
        from tensorflow.keras.layers import Dense, ReLU
        x = Dense(self.net_dim)(x)
        x = ReLU()(x)
        for _ in range(self.num_layer - 1):
            x = Dense(self.net_dim)(x)
            x = ReLU()(x)
        return x

    def build_q_network(self, conv=None):
        """AMAF network: trunk -> value + H advantage heads + gate."""
        from tensorflow.keras import backend as K
        import tensorflow as tf
        from tensorflow.keras.layers import (Dense, Input, Lambda, ReLU,
                                              Softmax)
        from tensorflow.keras.models import Model

        input_shape = ((self.seq_len, self.observ_size)
                       if self.recurrent else (self.observ_size,))
        x_in = Input(shape=input_shape)
        h = self._trunk(x_in)

        # Value stream (same shape as Zhang's dueling value stream).
        v = Dense(self.net_dim, activation="relu")(h)
        v = Dense(1, activation="linear")(v)  # [B, 1]

        # H advantage heads, each shaped like Zhang's dueling advantage
        # stream: Dense(net_dim, relu) -> Dense(action_shape, linear).
        head_outs = []
        for _ in range(self.n_heads):
            a = Dense(self.net_dim, activation="relu")(h)
            a = Dense(self.action_shape, activation="linear")(a)
            head_outs.append(a)  # each [B, action_shape]

        # Gate: state-dependent softmax weights over heads.
        if self.gate_mode == "uniform":
            from tensorflow import constant
            router = constant(
                [[1.0 / self.n_heads] * self.n_heads])
            fused = Lambda(
                lambda hs: sum(w * a for w, a in
                               zip([1.0 / self.n_heads] * self.n_heads,
                                   hs))
            )(head_outs)
        else:
            gate_logits = Dense(self.n_heads, activation="linear")(h)
            router = Softmax()(gate_logits)  # [B, H]

            def _fuse(hs, r):
                # hs: list of [B, A]; r: [B, H]
                stacked = tf.stack(hs, axis=1)  # [B, H, A]
                return tf.reduce_sum(
                    tf.expand_dims(r, -1) * stacked, axis=1)  # [B, A]

            fused = Lambda(lambda t: _fuse(t[0], t[1]))([head_outs, router])

        if self.fusion_mode == "uniform_anchored" and self.n_heads > 1:
            # tau(s): state-dependent trust in the adaptive fusion,
            # sigmoid-parameterized and initialized at trust_init.
            t_logit = Dense(self.net_dim, activation="relu")(h)
            t_logit = Dense(1, activation="linear")(t_logit)
            # bias init so that sigmoid(logit) == trust_init at t=0
            import math as _math
            b0 = _math.log(self.trust_init / (1.0 - self.trust_init))
            from tensorflow.keras import initializers
            t_dense = Dense(1, activation="sigmoid",
                            kernel_initializer="zeros",
                            bias_initializer=initializers.Constant(b0))
            tau = t_dense(h)  # [B, 1]

            from tensorflow import reduce_mean as _rmean

            def _anchor(t):
                hs, r, tau_v = t
                stacked = tf.stack(hs, axis=1)              # [B,H,A]
                uniform = tf.reduce_mean(stacked, axis=1)   # [B,A]
                adaptive = tf.reduce_sum(
                    tf.expand_dims(r, -1) * stacked, axis=1)  # [B,A]
                return uniform + tau_v * (adaptive - uniform)

            fused = Lambda(_anchor)([head_outs, router, tau])

        # Dueling combination: Q = V + A - mean(A)
        def _dueling(t):
            v_v, a_v = t
            return v_v + a_v - tf.reduce_mean(a_v, axis=1, keepdims=True)

        output = Lambda(_dueling, output_shape=(self.action_shape,))(
            [v, fused])

        model = Model(inputs=x_in, outputs=output)
        return model


class AMAF:
    """Controller wrapper: same public interface as agent.dqn.DQN."""

    def __init__(self, observ_space, action_shape, args=None,
                 act_only=False):
        self.name = "AMAF"
        self.model_dir = args.cwd

        self.if_recurrent = getattr(args, "if_recurrent", False)
        self.n_agents = getattr(args, "n_agents", 1)
        self.state_shape = getattr(args, "state_shape", 10)
        self.observ_space = observ_space
        self.action_shape = action_shape
        self.if_mac = getattr(args, "if_mac", False)
        output_size = (sum(self.action_shape) if self.if_mac
                      else self.action_shape)
        input_size = self.state_shape if self.if_mac else self.observ_space

        self.seq_len = (getattr(args, "seq_len", 3)
                        if self.if_recurrent else None)
        self.graph_conv = getattr(args, "global_state", False)
        self.agent = AMAFAgent(output_size, input_size, args,
                               self.seq_len, self.graph_conv)

        self.action_table = getattr(args, "action_table", None)
        self.if_norm = getattr(args, "if_norm", False)
        if self.if_norm:
            self.state_norm = array(
                [[i for _ in range(self.state_shape)] for i in range(2)])
        self.epsilon = getattr(args, "epsilon", 1)

        if not act_only:
            self.double = getattr(args, "if_double", True)
            self.gamma = getattr(args, "gamma", 0.98)
            self.batch_size = getattr(args, "batch_size", 256)
            self.learning_rate = getattr(args, "learning_rate", 1e-5)
            self.repeat_times = getattr(args, "repeat_times", 2)
            self.update_interval = getattr(args, "update_interval", 0.005)
            self.target_update_func = (
                self._hard_update_target_model
                if self.update_interval > 1
                else self._soft_update_target_model)
            self.episode = getattr(args, "episode", 0)

            self.trainable_variables = []
            self.target_trainable_variables = []
            self.trainable_variables += self.agent.model.trainable_variables
            self.target_trainable_variables += (
                self.agent.target_model.trainable_variables)

            self.loss_fn = ks.losses.get(args.loss_function)
            self.optimizer = ks.optimizers.get(args.optimizer)
            self.optimizer.learning_rate = self.learning_rate

        if args.if_load:
            self.load()

    # ------------------------------------------------------------------
    # Acting
    #2 ----------------------------------------------------------------------
    def act(self, state, train=True):
        if train and random.random() < self.epsilon:
            if self.if_mac:
                action = [random.randint(0, shape - 1)
                          for shape in self.action_shape]
            else:
                action = (random.randint(0, self.action_shape - 1),)
        else:
            if self.if_norm:
                state = self._normalize_state(state)
            state = expand_dims(convert_to_tensor(state), 0)
            a = self.agent.act(state)
            if self.if_mac:
                action = []
                for i, shape in enumerate(self.action_shape):
                    j = sum(self.action_shape[:i])
                    action.append(argmax(a[j:j + shape]))
            else:
                action = (argmax(a),)
        return action

    def convert_action_to_setting(self, action):
        if self.action_table is not None:
            setting = self.action_table[tuple(action)]
            return setting
        else:
            setting = [int(act) for act in action]
            return setting

    # ------------------------------------------------------------------
    # Learning
    # ----------------------------------------------------------------------
    def update_net(self, memory, batch_size=None):
        if self.if_norm:
            self.state_norm = memory.get_state_norm()
        batch_size = self.batch_size if batch_size is None else batch_size
        update_times = (int(1 + 4 * len(memory) / memory.limit)
                        * self.repeat_times)
        losses = []
        for _ in range(update_times):
            s, a, r, s_, d = memory.sample(batch_size)
            if self.if_norm:
                s, s_ = self._normalize_state(s), self._normalize_state(s_)
            loss = self._experience_replay(s, a, r, s_, d)
            self.target_update_func()
            losses.append(loss)
        return losses

    def evaluate_net(self, trajs):
        s, a, r, s_, d = [[traj[i] for traj in trajs] for i in range(5)]
        if self.if_norm:
            s, s_ = self._normalize_state(s), self._normalize_state(s_)
        loss = self._test_loss(s, a, r, s_, d)
        return loss

    def _normalize_state(self, s):
        s = ((array(s) - self.state_norm[0, :])
             / (self.state_norm[1, :] + 1e-5)).tolist()
        return s

    def _experience_replay(self, s, a, r, s_, d):
        s, r, s_, d = [convert_to_tensor(i, dtype=float32)
                       for i in [s, r, s_, d]]
        a = convert_to_tensor(a)
        if not self.if_mac:
            a = squeeze(a)
        targets = self._calculate_target(r, s_, d)
        loss = self._train_on_batch(s, a, targets)
        return loss

    def _train_on_batch(self, s, a, targets):
        with GradientTape() as tape:
            tape.watch(self.trainable_variables)
            y_preds = self.agent.forward(s)
            if self.if_mac:
                y_preds = [reduce_sum(
                    y_preds[:, sum(self.action_shape[:i]):
                            sum(self.action_shape[:i]) + shape]
                    * one_hot(a[:, i], shape), axis=1)
                    for i, shape in enumerate(self.action_shape)]
                y_preds = reduce_sum(convert_to_tensor(y_preds), axis=0)
            else:
                y_preds = reduce_sum(
                    y_preds * one_hot(a, depth=self.action_shape), axis=1)
            loss_value = self.loss_fn(targets, y_preds)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))
        return loss_value.numpy()

    def _calculate_target(self, r, s_, d):
        if self.double:
            if self.if_mac:
                argmax_actions = [ks.backend.argmax(
                    self.agent.forward(s_)[:, sum(self.action_shape[:i]):
                                          sum(self.action_shape[:i]) + shape])
                    for i, shape in enumerate(self.action_shape)]
                target_q_values = self.agent.forward(s_, target=True)
                target_q_values = [reduce_sum(
                    target_q_values[:, sum(self.action_shape[:i]):
                                   sum(self.action_shape[:i]) + shape]
                    * one_hot(argmax_actions[i], shape), axis=1)
                    for i, shape in enumerate(self.action_shape)]
                target_q_values = reduce_sum(
                    convert_to_tensor(target_q_values), axis=0)
            else:
                argmax_actions = ks.backend.argmax(
                    self.agent.forward(s_))
                target_q_values = reduce_sum(
                    self.agent.forward(s_, target=True)
                    * one_hot(argmax_actions, self.action_shape), axis=1)
        else:
            target_q_values = reduce_max(
                self.agent.forward(s_, target=True), axis=1)
        discounted_reward_batch = self.gamma * target_q_values
        targets = r + discounted_reward_batch * (1 - d)
        return targets

    def _test_loss(self, s, a, r, s_, d):
        s, r, s_, d = [convert_to_tensor(i, dtype=float32)
                       for i in [s, r, s_, d]]
        a = convert_to_tensor(a)
        if not self.if_mac:
            a = squeeze(a)
        targets = self._calculate_target(r, s_, d)
        y_preds = self.agent.forward(s)
        if self.if_mac:
            y_preds = [reduce_sum(
                y_preds[:, sum(self.action_shape[:i]):
                        sum(self.action_shape[:i]) + shape]
                * one_hot(a[:, i], shape), axis=1)
                for i, shape in enumerate(self.action_shape)]
            y_preds = reduce_sum(convert_to_tensor(y_preds), axis=0)
        else:
            y_preds = reduce_sum(
                y_preds * one_hot(a, depth=self.action_shape), axis=1)
        loss_value = self.loss_fn(targets, y_preds)
        return loss_value.numpy()

    def episode_update(self, episode, epsilon):
        self.episode = episode
        self.epsilon = epsilon

    def _hard_update_target_model(self):
        if self.episode % self.update_interval == 0:
            self.agent._hard_update_target_model()

    def _soft_update_target_model(self):
        self.agent._soft_update_target_model()

    # ------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------
    def save(self, model_dir=None, norm=True, agents=True):
        from numpy import save
        if norm and self.if_norm:
            d = self.model_dir if model_dir is None else model_dir
            save(join(d, "state_norm.npy"), self.state_norm)
        if agents:
            self.agent.save(0, model_dir)

    def load(self, model_dir=None, norm=True, agents=True):
        from numpy import load
        if norm and self.if_norm:
            d = self.model_dir if model_dir is None else model_dir
            self.state_norm = load(join(d, "state_norm.npy"))
        if agents:
            self.agent.load(0, model_dir)