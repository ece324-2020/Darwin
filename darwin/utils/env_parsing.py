def parse_kwargs(args):
    if not args: return None

    out = dict()
    for arg in args:
        k, v = arg.split('=')
        out[k] = int(v) if v.isnumeric() else v
    return out
