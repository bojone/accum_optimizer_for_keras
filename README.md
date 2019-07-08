[English|中文]

# 为Keras实现梯度累积版优化器 (accum_optimizer_for_keras)

继承Optimizer类，包装原有优化器，实现梯度累积功能。能够无缝对接原有优化器，不需要重写优化器。
