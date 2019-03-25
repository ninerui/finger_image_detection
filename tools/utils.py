def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        try:
            for key in args.__dir__():
                if not key.startswith("__") and not key.endswith("__"):
                    f.write('%s: %s\n' % (key, getattr(args, key)))
        except AttributeError:
            for key, value in vars(args).items():
                f.write('%s: %s\n' % (key, str(value)))


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1] == '-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
