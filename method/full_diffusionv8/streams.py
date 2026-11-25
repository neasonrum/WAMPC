from collections import defaultdict
from copy import deepcopy
import random

class DistributedRouteStreams:
    """Fixed disjoint route pools per stream; commit only consumed batches.

    Like the reference's SceneStreamBatchSampler, epochs are sample budgets and
    route streams continue across epochs. Prefetch never advances checkpoint state.
    """
    def __init__(self, samples, batch_size, rank=0, world_size=1, seed=901):
        groups = defaultdict(list)
        for index, (route, slot) in enumerate(samples):
            groups[int(route)].append((int(slot), index))
        self.scenes = [sorted(group) for _, group in sorted(groups.items())]
        self.batch_size, self.rank, self.world_size, self.seed = batch_size, rank, world_size, seed
        if len(self.scenes) < batch_size * world_size:
            raise ValueError('Need one distinct route pool per global stream')
        self.consumed, self.committed, self.pending = 0, None, {}

    def initial(self):
        order = list(range(len(self.scenes)))
        random.Random(self.seed).shuffle(order)
        total = self.batch_size * self.world_size
        return [{'pool': order[(self.rank * self.batch_size + i)::total], 'route': 0, 'frame': 0,
                 'cycle': 0, 'global_slot': self.rank * self.batch_size + i} for i in range(self.batch_size)]

    def __iter__(self):
        streams, cursor = deepcopy(self.committed if self.committed is not None else self.initial()), self.consumed
        while True:
            batch = []
            for stream in streams:
                if stream['route'] == len(stream['pool']):
                    stream['cycle'] += 1
                    random.Random(self.seed + 100003 * stream['cycle'] + stream['global_slot']).shuffle(stream['pool'])
                    stream['route'] = stream['frame'] = 0
                sequence = self.scenes[stream['pool'][stream['route']]]
                batch.append(sequence[stream['frame']][1])
                stream['frame'] += 1
                if stream['frame'] == len(sequence):
                    stream['route'] += 1
                    stream['frame'] = 0
            cursor += 1
            self.pending[cursor] = deepcopy(streams)
            yield batch

    def commit(self):
        self.consumed += 1
        self.committed = self.pending.pop(self.consumed)

    def state_dict(self):
        return {'consumed': self.consumed, 'streams': deepcopy(self.committed),
                'identity': [self.batch_size, self.rank, self.world_size, self.seed]}

    def load_state_dict(self, value):
        if value['identity'] != [self.batch_size, self.rank, self.world_size, self.seed]:
            raise ValueError('Stream resume requires identical batch, rank, world size and seed')
        self.consumed, self.committed = value['consumed'], deepcopy(value['streams'])
        self.pending = {}

