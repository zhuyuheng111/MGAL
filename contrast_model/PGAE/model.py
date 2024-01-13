from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, Fcn
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)


class GCNModelPAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelPAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        # 1st gcn layer
        self.hidden11 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden11,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.hidden21 = GraphConvolutionSparse(input_dim=self.input_dim,
                                               output_dim=FLAGS.hidden21,
                                               adj=self.adj,
                                               features_nonzero=self.features_nonzero,
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               logging=self.logging)(self.inputs)

        self.hidden31 = GraphConvolutionSparse(input_dim=self.input_dim,
                                               output_dim=FLAGS.hidden31,
                                               adj=self.adj,
                                               features_nonzero=self.features_nonzero,
                                               act=tf.nn.relu,
                                               dropout=self.dropout,
                                               logging=self.logging)(self.inputs)


        # 2th gcn layer
        self.hidden12 = GraphConvolution(input_dim=FLAGS.hidden11,
                                           output_dim=FLAGS.hidden12,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden11)

        self.hidden22 = GraphConvolution(input_dim=FLAGS.hidden21,
                                         output_dim=FLAGS.hidden22,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(self.hidden21)

        self.hidden32 = GraphConvolution(input_dim=FLAGS.hidden31,
                                         output_dim=FLAGS.hidden32,
                                         adj=self.adj,
                                         act=lambda x: x,
                                         dropout=self.dropout,
                                         logging=self.logging)(self.hidden31)

        # fcn layer
        self.embeddding1 = Fcn(input_dim=FLAGS.hidden12,
                               output_dim=128,
                               act=lambda x: x,
                               dropout=self.dropout,
                               logging=self.logging)(self.hidden12)

        self.embeddding2 = Fcn(input_dim=FLAGS.hidden22,
                               output_dim=128,
                               act=lambda x: x,
                               dropout=self.dropout,
                               logging=self.logging)(self.hidden22)

        self.embeddding3 = Fcn(input_dim=FLAGS.hidden32,
                               output_dim=128,
                               act=lambda x: x,
                               dropout=self.dropout,
                               logging=self.logging)(self.hidden32)

        self.z_mean = (self.embeddding1 + self.embeddding2 + self.embeddding3) / 3


        self.reconstructions = InnerProductDecoder(input_dim=self.input_dim,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.z_mean)
