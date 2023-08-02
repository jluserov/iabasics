#https://weehourstechnology.com/adding-two-numbers-using-keras-machine-learning/
from keras.models import Sequential, load_model
from keras.layers import Dense
from random import seed, randint
from numpy import array
from fractions import Fraction


def create_triples(count, max_value):
    addends = list()
    sums = list()
    for n in range(count):
        addends.append([randint(0, max_value), randint(0, max_value)])
        sums.append(sum(addends[n]))
    addends = array(addends)
    sums = array(sums)
    #print(addends)
    #print(sums)
    return addends, sums


def denormalize(value, maxvalue):
    return value * float(maxvalue * 2.0)


def normalize(value, maxvalue):
    return value.astype('float') / float(maxvalue * 2.0)


def setup_model(m):
    m.add(Dense(4, input_dim=2))
    m.add(Dense(2))
    m.add(Dense(1))
    m.compile(loss='mean_squared_error', optimizer='adam')


def train_model(m):
    for _ in range(50):
        x, y = create_triples(valueCount, maxValue)
        x2 = normalize(x, maxValue)
        y2 = normalize(y, maxValue)
        m.fit(x2, y2, epochs=3, batch_size=2, verbose=0)


def save_model(m, filename):
    m.save(filename)


# ----------------------------------------------------------
seed(100)
valueCount: int = 100
maxValue: int = 100
model = Sequential()

# decide whether to load/use existing model or retrain and save changed model
use_existing_model = False

if use_existing_model:
    model = load_model('trained_model_2.h5')
else:
    setup_model(model)
    train_model(model)
    model.save('trained_model_2.h5')

# evaluate model
x, y = create_triples(count=20, max_value=maxValue)
x2 = normalize(x, maxValue)
testresult = model.predict(x2, batch_size=1, verbose=0)

# show results
for i in range(len(testresult)):
    addend = denormalize(x2[i][0], maxValue)
    augend = denormalize(x2[i][1], maxValue)
    total = denormalize(testresult[i][0], maxValue)
    print('{:4d} {:12.6f} {:12.6f} {:12.6f} {:4d}'.format(i, addend, augend, total, y[i]))

print("\r\n")

# do predictions of hand-coded values
x = [[1200, 1343], [1, 1], [-3, -3], [Fraction(16, 5), 3.25]]
x = array(x)
x2 = normalize(x, maxValue)
testresult = model.predict(x2, batch_size=1, verbose=0)

# show results
for i in range(len(testresult)):
    addend = denormalize(x2[i][0], maxValue)
    augend = denormalize(x2[i][1], maxValue)
    total = denormalize(testresult[i][0], maxValue)
    print('{:4d} {:12.6f} {:12.6f} {:12.6f} {:8.2f}'.format(i, addend, augend, total, total))

exit(0)
