# Distributed-and-Local-Logistic-Regression-with-TensorFlow
AIM: To implement and evaluate a L2-regularized logistic regression in the standalone and distributed setting using Parameter Server.
preprocessing.py  

For running local mode : python local_train.py

For Distributed(Change ip for ps and worker nodes in codes)

Bulk Synchronous Parallel mode
on ps node : python bsp.py --job_name="ps" --task_index=0

on worker0 node: python bsp.py --job_name="worker" --task_index=0

on worker 1 node: python bsp.py --job_name="worker" --task_index=1

Asynchronous Parallel mode
on ps node : python async.py --job_name="ps" --task_index=0

on worker0 node: python async.py --job_name="worker" --task_index=0

on worker 1 node: python async.py --job_name="worker" --task_index=1

Stale Synchronus Parallel mode(change stale value as required)
on ps node : python ssp.py --stale=32 --job_name="ps" --task_index=0

on worker0 node: python ssp.py --stale=32 --job_name="worker" --task_index=0

on worker 1 node: python ssp.py --stale=32 --job_name="worker" --task_index=1
