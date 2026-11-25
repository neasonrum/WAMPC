"""Lossless values, dtypes and non-overlapping strides across local IPC."""
import pickle
from pathlib import Path
from multiprocessing.connection import Client


def pack(value):
    import torch
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        bf16 = cpu.dtype == torch.bfloat16
        return {'__tensor__': True, 'array': (cpu.view(torch.uint16) if bf16 else cpu).numpy(),
                'bf16': bf16, 'stride': tuple(value.stride())}
    if isinstance(value, tuple):
        return tuple(pack(x) for x in value)
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value, device='cpu'):
    import torch
    if isinstance(value, dict) and value.get('__tensor__') is True:
        tensor = torch.from_numpy(value['array'])
        if value['bf16']:
            tensor = tensor.view(torch.bfloat16)
        strides = value['stride']
        if any(s == 0 and n > 1 for s, n in zip(strides, tensor.shape)):
            return tensor.to(device)
        output = torch.empty_strided(tensor.shape, strides, dtype=tensor.dtype, device=device)
        output.copy_(tensor)
        return output
    if isinstance(value, tuple):
        return tuple(unpack(x, device) for x in value)
    if isinstance(value, list):
        return [unpack(x, device) for x in value]
    if isinstance(value, dict):
        return {k: unpack(v, device) for k, v in value.items()}
    return value


def send(connection, value):
    connection.send_bytes(pickle.dumps(value, protocol=5))


def receive(connection):
    return pickle.loads(connection.recv_bytes(64 * 1024 * 1024))


class PolicyClient:
    def __init__(self, address, token):
        self.connection = Client(address, family='AF_UNIX', authkey=Path(token).read_bytes())

    def call(self, operation, **values):
        send(self.connection, {'operation': operation, **values})
        if not self.connection.poll(240):
            raise TimeoutError('Shared point-query inference exceeded 240 seconds')
        result = receive(self.connection)
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result['value']

    def close(self):
        try:
            self.call('close')
        finally:
            self.connection.close()

