
# BUG LOG
运行 dist_test.sh / test_mmagic.py 的时候，会遇到以下问题：
```shell
"The model and loaded state dict do not match exactly

unexpected key in source state_dict: generator.module.conv1.weight, generator.module.conv1.bias, generator.module.conv2.weight, generator.module.conv2.bias, generator.module.conv3.weight, generator.module.conv3.bias

missing keys in source state_dict: generator.conv1.weight, generator.conv1.bias, generator.conv2.weight, generator.conv2.bias, generator.conv3.weight, generator.conv3.bias"
```

问题分析：
- 首先，你会怀疑网络模型和保存的 checkpoint 中的 state_dict 不匹配，其实并不是的；
- 这主要是因为 mmengine.runner.chechpoint.py 中的 函数 "_load_checkpoint_to_model" (大概 585 行) 的参数 revise_keys=[(r'^module//.', '')]: 有问题；
- '^module//.' 是 re 中的一种模式，想将类似 "generator.module.conv1.weight" 改为 "generator.conv1.weight" ，即去掉 "generator.module.conv1.weight" 中的 "module."；
- 但是，由于 "generator.module.conv1.weight" 并不是以 "module." 开头，即不匹配模式 '^module//.'；

解决方法：
- mmengine.runner.runner.py 中类 Runner 的函数 load_checkpoint 中的参数 "revise_keys=[(r'^module//.','')]):"(大概 2111 行) 替换为"revise_keys=[(r'/bmodule.', '')]):"，实现将类似 "generator.module.conv1.weight" 改为 "generator.conv1.weight" ，即去掉 "generator.module.conv1.weight" 中的 "module."!
