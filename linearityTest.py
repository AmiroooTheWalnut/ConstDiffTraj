import tensorflow as tf

# Simple score model: just a Dense layer
class SimpleScoreModel(tf.keras.Model):
    def __init__(self, input_dim):
        super().__init__()
        self.dense = tf.keras.layers.Dense(input_dim)

    def call(self, x, t):
        # Concatenate time embedding to input (you can customize this)
        t_embed = tf.cast(t, tf.float32)
        t_embed = tf.expand_dims(t_embed, -1)
        t_embed = tf.broadcast_to(t_embed, tf.shape(x))
        x_in = tf.concat([x, t_embed], axis=-1)
        return self.dense(x_in)

# Diffusion model using tf.scan
class DiffusionModel(tf.keras.Model):
    def __init__(self, score_model, num_steps, beta=0.01):
        super().__init__()
        self.score_model = score_model
        self.num_steps = num_steps
        self.beta = beta

    def call(self, x0):
        batch_size, input_dim = tf.shape(x0)[0], tf.shape(x0)[1]

        def diffusion_step(x_t, t):
            noise = tf.random.normal(tf.shape(x_t))
            score = self.score_model(x_t, t)
            x_t_next = x_t + self.beta * score + tf.sqrt(2 * self.beta) * noise
            return x_t_next

        time_steps = tf.range(self.num_steps, dtype=tf.int32)

        # Use tf.scan to apply diffusion_step over time steps
        def scan_fn(prev_x, t):
            return diffusion_step(prev_x, t)

        # Initial input to scan is x0
        xts = tf.scan(scan_fn, time_steps, initializer=x0)

        # xts is [num_steps, batch_size, input_dim], return final x_T
        return xts[-1]

# Example usage
input_dim = 4
batch_size = 2
num_steps = 100

score_model = SimpleScoreModel(input_dim=input_dim + 1)
diffusion = DiffusionModel(score_model, num_steps)

x0 = tf.random.normal((batch_size, input_dim))
x_T = diffusion(x0)

print("Final x_T shape:", x_T.shape)