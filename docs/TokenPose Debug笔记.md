# TokenPose Debug笔记



## nori2

```shell
ERROR: Could not find a version that satisfies the requirement nori2 (from versions: none)
ERROR: No matching distribution found for nori2
```

+ 直接注释掉`lib/dataset/JointsDataset.py`：

```python
# import nori2 as nr
```



## name 'nr' is not defined

```shell
Traceback (most recent call last):
  File "tools/train_custom.py", line 232, in <module>
    main()
  File "tools/train_custom.py", line 191, in main
    train(cfg, train_loader, model, criterion, optimizer, epoch,
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/core/function.py", line 40, in train
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 521, in __next__
    data = self._next_data()
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1203, in _next_data
    return self._process_data(data)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1229, in _process_data
    data.reraise()
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/_utils.py", line 425, in reraise
    raise self.exc_type(msg)
NameError: Caught NameError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py", line 287, in _worker_loop
    data = fetcher.fetch(index)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 44, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 44, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/dataset/JointsDataset.py", line 119, in __getitem__
    self.fn = nr.Fetcher()
NameError: name 'nr' is not defined
```

+ 将`lib/dataset/JointsDataset.py`的数据处理方式改回到TransPose等的处理方式：

```python
    def __getitem__(self, idx):
        """
        if self.fn is None:
            self.fn = nr.Fetcher()
        """

        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        """
        if 'nori_id' in db_rec.keys():
            nori_id = db_rec['nori_id']
            ns = np.fromstring(self.fn.get(nori_id), dtype=np.uint8)
            data_numpy = cv2.imdecode(ns, cv2.IMREAD_COLOR)
        else:
            if self.data_format == 'zip':
                from utils import zipreader
                data_numpy = zipreader.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )
            else:
                data_numpy = cv2.imread(
                    image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
                )
        """

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
```



## PoseMobileNet

```shell
Traceback (most recent call last):
  File "tools/train_custom.py", line 232, in <module>
    main()
  File "tools/train_custom.py", line 93, in main
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/models/pose_tokenpose_b.py", line 58, in get_pose_net
    model = TokenPose_B(cfg, **kwargs)
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/models/pose_tokenpose_b.py", line 31, in __init__
    super(PoseMobileNet, self).__init__()
NameError: name 'PoseMobileNet' is not defined
```

+ 修改`lib/models/pose_tokenpose_b.py`、`lib/models/pose_tokenpose_s.py`、`lib/models/pose_tokenpose_t.py`等3个文件的bug

```python
super(PoseMobileNet, self).__init__()
```



## CUDA error: out of memory

```shell
Traceback (most recent call last):
  File "tools/train_custom.py", line 232, in <module>
    main()
  File "tools/train_custom.py", line 93, in main
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/models/pose_tokenpose_b.py", line 60, in get_pose_net
    model.init_weights(cfg.MODEL.PRETRAINED)
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/models/pose_tokenpose_b.py", line 54, in init_weights
    self.pre_feature.init_weights(pretrained)
  File "/opt/SRC/projects/keypoint_detection/TokenPose/tools/../lib/models/hr_base.py", line 470, in init_weights
    pretrained_state_dict = torch.load(pretrained)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/serialization.py", line 608, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/serialization.py", line 787, in _legacy_load
    result = unpickler.load()
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/serialization.py", line 743, in persistent_load
    deserialized_objects[root_key] = restore_location(obj, location)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/serialization.py", line 175, in default_restore_location
    result = fn(storage, location)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/serialization.py", line 155, in _cuda_deserialize
    return storage_type(obj.size())
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/cuda/__init__.py", line 528, in _lazy_new
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```



## CUDA error: out of memory

```shell
Traceback (most recent call last):
  File "tools/train_custom.py", line 232, in <module>
    main()
  File "tools/train_custom.py", line 117, in main
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in cuda
    return self._apply(lambda t: t.cuda(device))
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
    module._apply(fn)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
    module._apply(fn)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 530, in _apply
    module._apply(fn)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 552, in _apply
    param_applied = fn(param)
  File "/opt/Software/miniconda3/envs/tokenpose/lib/python3.8/site-packages/torch/nn/modules/module.py", line 637, in <lambda>
    return self._apply(lambda t: t.cuda(device))
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

