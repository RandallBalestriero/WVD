import pickle
import numpy as np
import symjax
import symjax.tensor as T
import argparse
from wvd import data_loader, utils
import numpy_datasets as nds


parse = argparse.ArgumentParser()
parse.add_argument(
    "--option",
    type=str,
    choices=[
        "melspec",
        "morlet",
        "sinc",
        "learnmorlet",
        "raw",
        "rawshort",
        "wvd",
        "mwvd",
    ],
)
parse.add_argument("-J", type=int, default=5)
parse.add_argument("-Q", type=int, default=8)
parse.add_argument("--bins", type=int, default=1024)
parse.add_argument("-BS", type=int, default=16)
parse.add_argument("-L", type=int, default=0)
parse.add_argument("-lr", type=float, default=0.001)
parse.add_argument("--wvdinit", type=str, default="gabor")
parse.add_argument(
    "--model",
    type=str,
    choices=[
        "onelayer_nonlinear_scattering",
        "onelayer_linear_scattering",
        "joint_linear_scattering",
        "joint_nonlinear_scattering",
        "deep_net",
    ],
)
parse.add_argument("--hop", type=int, default=0)
parse.add_argument("--epochs", type=int, default=100)
parse.add_argument("--dataset", type=str)
parse.add_argument("--modes", type=int, default=1)
args = parse.parse_args()

if args.hop == 0:
    args.hop = args.bins // 4


# DATASET LOADING
(
    wavs_train,
    labels_train,
    wavs_valid,
    labels_valid,
    wavs_test,
    labels_test,
    Y,
) = data_loader.get(args.dataset)

print("dataset loaded")
print(Y)

# COMPUTATIONAL MODEL

# create placeholders
if args.dataset == "usc":
    label = T.Placeholder((args.BS, Y), "int32")
else:
    label = T.Placeholder((args.BS,), "int32")

input = T.Placeholder((args.BS, len(wavs_train[0])), "float32")
deterministic = T.Placeholder((), "bool")

print("before transform")
print(input)
tf = utils.transform(input, args)
output = utils.__dict__[args.model](tf, deterministic, Y)
print("after transform")

# create loss function and loss
if args.dataset == "usc":
    loss = symjax.losses.sigmoid_crossentropy_logits(label, output).mean()
    indices = T.greater_equal(output, 0).astype("int")
    accuracy = T.equal(label, indices).astype("float32").mean(0)
else:
    loss = symjax.nn.losses.sparse_softmax_crossentropy_logits(
        label, output
    ).mean()
    accuracy = symjax.nn.losses.accuracy(label, output).mean()

print("before adam")
symjax.nn.optimizers.Adam(loss, args.lr)
print("after adam")
# create the functions
train = symjax.function(
    input, label, deterministic, outputs=loss, updates=symjax.get_updates()
)
print("created train function")
test = symjax.function(input, label, deterministic, outputs=[loss, accuracy])
get_repr = symjax.function(input, outputs=tf)

filename = "data/resave_{}_{}_{}_{}_{}_"

filename = filename.format(
    args.option, args.wvdinit, args.model, args.dataset, args.lr,
)
for run in range(10):
    TRAIN, TEST, VALID, FILTER, REP = [], [], [], [], []
    symjax.reset_variables()

    for epoch in range(args.epochs):

        # train part
        l = list()
        for xx, xy in nds.utils.batchify(
            wavs_train,
            labels_train,
            batch_size=args.BS,
            option="random_see_all",
        ):
            print("starts!")
            l.append(train(xx, xy, 0))
            print(l[-1])
        print("FINALtrain", np.mean(l))
        TRAIN.append(np.array(l))

        # valid and get repr
        l = list()
        for x, y in nds.utils.batchify(
            wavs_valid, labels_valid, batch_size=args.BS, option="continuous"
        ):
            l.append(test(x, y, 1))

        VALID.append(np.array(l).mean(0))
        print("FINALvalid", VALID[-1])

        # test
        l = list()
        for x, y in nds.utils.batchify(
            wavs_test, labels_test, batch_size=args.BS, option="continuous"
        ):
            l.append(test(x, y, 1))

        TEST.append(np.array(l).mean(0))
        print("FINALtest", TEST[-1])

        # save the file
        # if epoch == 0 or epoch == args.epochs - 1:
        REP.append(get_repr(wavs_train[: args.BS]))
        # save filter parameters
        # if "wvd" in args.option:
        #     FILTER.append(
        #         [
        #             layer[0]._mu.get({}),
        #             layer[0]._sigma.get({}),
        #             layer[0]._cor.get({}),
        #             layer[0]._filter.get({}),
        #         ]
        #     )
        # elif "sinc" == args.option:
        #     FILTER.append(
        #         [layer[0]._freq.get({}), layer[0]._filter.get({})]
        #     )
        # elif "learnmorlet" == args.option:
        #     FILTER.append(
        #         [
        #             layer[0]._filters.get({}),
        #             layer[0]._freqs.get({}),
        #             layer[0]._scales.get({}),
        #         ]
        #     )

        np.savez(
            filename + str(run) + ".npz",
            train=TRAIN,
            test=TEST,
            valid=VALID,
            filter=FILTER,
            rep=REP,
            wavs=wavs_train[: args.BS],
        )
