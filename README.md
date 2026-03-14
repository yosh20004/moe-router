# moe-router

### what is this repo for?
- high performance cuda kernel for moe router
- reference from [nvidia transformer engine](https://github.com/NVIDIA/TransformerEngine)

### how to build
- for cuda
  ```bash
  python setup.py build_ext --inplace
  ```
- for musa
  ```
  coming soon
  ```

### accuracy test
```bash
python -m pytest tests/test_fused_router.py -s -q
```

### performance test
```bash
python -m pytest tests/test_fused_router_perf.py -s -q
```


### todo
- [ ] support musa