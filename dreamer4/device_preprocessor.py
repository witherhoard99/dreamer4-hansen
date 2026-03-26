import torch
import collections


class DevicePreprocessor:
    def __init__(self, loader, device, preprocess_fn, device_preprocess_factor=2):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.device_preprocess_factor = device_preprocess_factor
        self.preprocess_fn = preprocess_fn
        self.queue = collections.deque()

    def __iter__(self):
        self.queue.clear()

        loader_it = iter(self.loader)

        for _ in range(self.device_preprocess_factor):
            self.preload(loader_it)

        while len(self.queue) > 0:
            batch = self.queue.popleft()

            torch.cuda.current_stream().wait_stream(self.stream)

            batch = self.record_stream(batch)
            yield batch
            self.preload(loader_it)

    def preload(self, it):
        try:
            batch = next(it)
        except StopIteration:
            return

        with torch.cuda.stream(self.stream):
            batch = self._move_to_device(batch)

            if self.preprocess_fn:
                batch = self.preprocess_fn(batch)

        self.queue.append(batch)

    def _move_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            # Preserve the type (tuple vs list)
            return type(batch)(self._move_to_device(x) for x in batch)
        return batch

    def record_stream(self, batch):
        if isinstance(batch, torch.Tensor):
            batch.record_stream(torch.cuda.current_stream())
        elif isinstance(batch, (list, tuple)):
            for x in batch:
                self.record_stream(x)  # Recursive call to be safe
        return batch
