import numpy as np
import tensorflow as tf

def positional_encoding(length: int, depth: int):
    '''
    
    '''
    half_depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(half_depth)[np.newaxis, :]/half_depth
    angle_rates = 1/(10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    '''
    
    '''
    def __init__(self, vocab_size: int, max_seq_len: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 2,
            output_dim=hidden_dim,
            input_length=max_seq_len,
            mask_zero=True
        )
        self.positional_encoding = positional_encoding(
            length=max_seq_len,
            depth=hidden_dim
        )

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.hidden_dim, tf.float32))
        x += self.positional_encoding[tf.newaxis, :length, :]
        return x
    
class GlobalAttention(tf.keras.layers.Layer):
    '''
    
    '''
    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=hidden_dim
        )        
        self.layernorm=tf.keras.layers.LayerNormalization()
        self.add=tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return(x)
    
class FeedForward(tf.keras.layers.Layer):
    '''
    
    '''
    def __init__(self, hidden_dim: int, dropout_pr: float):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim*4, activation='relu'),
            tf.keras.layers.Dense(hidden_dim),
            tf.keras.layers.Dropout(dropout_pr)
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layernorm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    '''
    
    '''
    def __init__(self, hidden_dim: int, num_heads: int, dropout_pr: float):
        super().__init__()
        self.self_attention = GlobalAttention(num_heads, hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout_pr)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(tf.keras.Model):
    '''
    
    '''
    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        max_seq_len: int,
        dropout_pr: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size, 
            max_seq_len, 
            hidden_dim
        )
        self.enc_layers = [
            EncoderLayer(hidden_dim, num_heads, dropout_pr)
            for _ in range(n_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_pr)
        self.dense_out = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(vocab_size,activation='softmax')
        )
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x_layers = []
        for i in range(self.n_layers):
            x = self.enc_layers[i](x)
            x_layers.append(x)
        y = self.pooling(x)    
        x = self.dense_out(x)
        return x, y, x_layers