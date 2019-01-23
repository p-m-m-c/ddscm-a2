class Celsius:
    def __init__(self, temperature = 0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self._temperature = value

a = Celsius(temperature=8)
print(a.temperature)
print(a._temperature)
a._temperature = -800 # We modify the internal variable
print(a.temperature)
print(a.to_fahrenheit()) # This now yields the incorrect result