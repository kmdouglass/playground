from logging import getLogger
from queue import Queue
import random
import time
from typing import Any, Iterable, Iterator

from pymmcore_plus import CMMCorePlus
from useq import MDAEvent, MDASequence


logger = getLogger(__name__)


class Analyzer:
    """Analyzes images and returns a dict of results."""
    def run(self, data) -> dict[str, Any]:
        # Fake analysis; randomly return a dict with a value of None 10% of the time
        if random.random() < 0.1:
            return {"result": None}
        else:
            return {"result": random.random()}


class Controller:
    STOP_EVENT = object()

    def __init__(self, analyzer: Analyzer, mmc: CMMCorePlus, queue: Queue):
        self._analyzer = analyzer
        self._mmc = mmc
        self._queue = queue

    def run(self):
        seq: Iterable[MDAEvent] = iter(self._queue.get, self.STOP_EVENT)
        self._mmc.run_mda(seq)  # Non-blocking

        event = MDAEvent(exposure=10)

        # Start the acquisition            
        self._queue.put(event)

        while True:
            # Perform a measurement
            # Normally, we'd have to wait on a new image event here, otherwise there might not be
            # anything in the buffer yet.
            time.sleep(.1)
            self._mmc.snapImage()
            img = self._mmc.getImage()

            # Analyze the image
            results = self._analyzer.run(img)

            # Decide what to do. This is the key part of the reactive loop.
            if results['result'] is None:
                # Do nothing and return
                logger.info("Analyzer returned no results. Stopping...")
                self._queue.put(self.STOP_EVENT)
                break
            else:
                # Adjust the exposure time based on the results and continue
                logger.info("Analyzer returned results. Continuing...")
                new_exp_time = 10 * results["result"]
                event = MDAEvent(exposure=new_exp_time)
                self._queue.put(event)


def main():
    q = Queue()

    # Setup the MM Core
    mmc = CMMCorePlus()
    mmc.loadSystemConfiguration()

    # Setup the controller and analyzer
    analyzer = Analyzer()
    controller = Controller(analyzer, mmc, q)

    # Start the acquisition
    controller.run()


if __name__ == "__main__":
    main()
