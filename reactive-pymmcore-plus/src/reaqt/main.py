from queue import Queue
from typing import Any, Iterable, Iterator

from pymmcore_plus import CMMCorePlus
from pymmcore_plus.mda import MDAEngine
from useq import MDAEvent


class Controller:
    def __init__(self, mmc: CMMCorePlus, queue: Queue):
        self._mmc = mmc
        self._queue = queue

    def run(self):
        seq: Iterable[MDAEvent] = iter(self._queue.get, None)
        self._mmc.mda.run(seq)
            

class ReaqtEngine(MDAEngine):
    def __init__(self, mmc: CMMCorePlus, queue: Queue):
        super().__init__(mmc)
        self._queue = queue

    def event_iterator(self, events: Iterable[MDAEvent]) -> Iterator[MDAEvent]:
        for event in events:
            yield event


class Analyzer:
    """Analyzes images and returns a dict of results."""
    def run(self, data) -> dict[str, Any]:
        pass



def main():
    q = Queue()

    # Setup the MM Core
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration()
    mmc.mda.set_engine(ReaqtEngine(mmc, q))

    # Setup the controller
    controller = Controller(mmc, q)

    controller.run()


if __name__ == "__main__":
    main()
