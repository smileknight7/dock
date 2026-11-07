


# autodock采用的数据处理方式都是mglt和autodock原生的方式


# 使用autodock原生脚本的时候是用dpf直接进行对接，其余使用map来进行对接，相对来讲比较方便，不用每个lig生成一个dpf文件
# autodock_gpu, autodockZn和autodock_vina都已经转向使用meeko/vina类似的方式了



# gpf并不是直接用于docking的文件，主要是网格参数配置文件，实际使用的是get_grid过程中的生成的map文件