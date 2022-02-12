def some_func():
    return (1,2)

_,a = some_func()








class computer():
    def __init__(self, system='windows'):
        self.system = system

    def activate(self):
        return 'booted!'

class mac(computer):
    # @ override
    def __init__(self, system='mac'):
        self.system = system
        self.price = 'high!'

win98 = computer(system='windows98')
win10 = computer(system='windows10')
print(win98.price)
print(win10.price)

mac12 = mac(system='macOS12')
print(mac12.price)
print(mac12.activate())