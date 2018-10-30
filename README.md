# Distributed-and-Local-Logistic-Regression-with-TensorFlow
AIM: To implement and evaluate a L2-regularized logistic regression in the standalone and distributed setting using Parameter Server.

To preprocess the dataset, kindly run preprocessing.py that will featurize input documents and output labels.  

To run in local mode : python local_implementation.py

Local implementation results could be seen in the .pynb file uploaded.

For Distributed mode:

ASP

on ps node : python asp.py --job_name="ps" --task_index=0

on worker0 node: python asp.py --job_name="worker" --task_index=0

on worker 1 node: python asp.py --job_name="worker" --task_index=1

BSP

on ps node : python bsp.py --job_name="ps" --task_index=0

on worker0 node: python bsp.py --job_name="worker" --task_index=0

on worker 1 node: python bsp.py --job_name="worker" --task_index=1

SSP(staleness value to be changed in ssp.py code accordingly)

on ps node : python ssp.py --stale=32 --job_name="ps" --task_index=0

on worker0 node: python ssp.py --stale=32 --job_name="worker" --task_index=0

on worker 1 node: python ssp.py --stale=32 --job_name="worker" --task_index=1
