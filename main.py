import logging

from Car import Car



def main():
    logging.basicConfig(format='%(name)-12s: %(asctime)s %(process)d %(levelname)-8s %(message)s', level=logging.DEBUG)
    with Car() as car:
    # car = Car()
        car.motorFL.throttle = None
        print(car.motorFL.throttle)
        # car.test_motor()
        car.drive() 
        
if __name__ == "__main__":
    main()