#!/usr/bin/env python

from jinja2 import Template

from settings import SOLVERS, RIA_MODEL_DIR


def render_solver(template, max_iter, snapshot, dst,
                  base_lr=0.01, gamma=0.96, stepsize=200,
                  momentum=0.9, weight_decay=0.0005,
                  solver_mode='CPU', test_interval=400, **options):
    data = template.render(
        base_lr=base_lr, gamma=gamma, stepsize=stepsize,
        momentum=momentum, weight_decay=weight_decay,
        max_iter=max_iter,
        snapshot=snapshot, solver_mode=solver_mode,
        test_interval=test_interval)

    solver = open(dst, 'w')
    solver.write(data)
    solver.close()


def main():
    base_dir = RIA_MODEL_DIR
    template = Template(open(base_dir + 'solver.prototxt.template', 'r').read())

    for options in SOLVERS:
        options['dst'] = base_dir+'%s_solver.prototxt' % options['name']

        render_solver(template, **options)


if __name__ == "__main__":
    main()