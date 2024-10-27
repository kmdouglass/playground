import random
import time

from disruptor import Consumer, Disruptor


class MyConsumer(Consumer):
    
    def __init__(self, name):
        self.name = name
    
    def consume(self, elements):
        # simulate some random processing delay
        time.sleep(random.random())
        print(f"{self.name} consumed {elements}")


def main() -> None:
    # Construct a couple of consumer instances
    consumer_one = MyConsumer(name="consumer one")
    consumer_two = MyConsumer(name="consumer two")

    disruptor = Disruptor(name="Example", size=3)
    try:
        # Register consumers
        disruptor.register_consumer(consumer_one)
        disruptor.register_consumer(consumer_two)

        for i in range(10):
            # Produce a bunch of elements
            element = f"element {i}"
            disruptor.produce([element])
            print(f"produced {element}")
    finally:
        # Shut down the disruptor
        disruptor.close()