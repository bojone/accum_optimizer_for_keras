[<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/README.md">中文</a>|<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/README_en.md">English</a>]

# 为Keras实现梯度累积版优化器

## 特点

继承Optimizer类，包装原有优化器，实现梯度累积功能。能够无缝对接原有优化器，不需要重写优化器。

## 用法

如下例子等价于直接使用batch_size=100的Adam优化器（代价就是你跑了10个epoch，实际上只相当于batch_size=100跑了1个epoch）：
```
opt = AccumOptimizer(Adam(), 10) # 10是累积步数
model.compile(loss='mse', optimizer=opt)
model.fit(x_train, y_train, epochs=10, batch_size=10)
```
读者也可以直接跑一跑<a href="https://github.com/bojone/accum_optimizer_for_keras/blob/master/mnist_mlp_example.py">mnist_mlp_example.py</a>。

## 链接
https://kexue.fm/archives/6794

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
