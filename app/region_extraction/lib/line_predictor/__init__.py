def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from .registry import MODULE_BUILD_FUNCS

    if args.modelname not in MODULE_BUILD_FUNCS._module_dict:
        build_model(args)

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def build_model(args):
    from .dino.dino import build_dino

    return build_dino(args)
