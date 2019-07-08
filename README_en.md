[<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/README.md">中文</a>|<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/README_en.md">English</a>]

# Keras Optimizer with Gradient Accumulation

## Feature

Inheriting Optimizer class, wrapping the original optimizer to achieve a new corresponding optimizer of gradient accumulation.

## Usage

The following example equals to use original Adam optimizer with batch_size=100 (but the cost is that you run 10 epoch under batch_size=10, which is actually equivalent to 1 epoch under batch_size=100):
```
opt = AccumOptimizer(Adam(), 10) # 10 is accumulative steps
model.compile(loss='mse', optimizer=opt)
model.fit(x_train, y_train, epochs=10, batch_size=10)
```
Or you can try<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/mnist_mlp_example.py">mnist_mlp_example.py</a> directly.

## Link
https://kexue.fm/archives/6794
