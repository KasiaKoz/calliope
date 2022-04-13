"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

constraints.py
~~~~~~~~~~

Create constraints for the model.

"""
import operator
import re
from calliope.backend.pyomo.util import get_param

SUPPORTED_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '**': operator.pow,
    '/': operator.truediv,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge
}
OPERATOR_HIERARCHY = ['*', '**', '/', '+', '-', '>', '<', '>=', '<=', '==', '!=']
OPERATOR_PATTERN = r'([^a-zA-Z0-9_ \,\]\[])'
NUMBER_PATTERN = r'([0-9])'
EXPRESSION_PATTERN = r'(\w+)'
FUNCTION_PATTERN = r'(\w+)\[\w+(, \w+)?\]'


def math_operation(operator: str, a, b):
    def template_operation(backend_model, **kwargs):
        if isinstance(a, float) and isinstance(b, float):
            return operate(a, b)
        elif isinstance(a, float):
            return operate(a, b(backend_model, **kwargs))
        elif isinstance(b, float):
            return operate(a(backend_model, **kwargs), b)
        return operate(a(backend_model, **kwargs), b(backend_model, **kwargs))
    if operator in SUPPORTED_OPERATORS:
        operate = SUPPORTED_OPERATORS[operator]
        return template_operation
    else:
        raise NotImplemented(f'Operator {operator} is not supported. Supported operators: {list(SUPPORTED_OPERATORS)}')


def _combine_consecutive_operators(l: list):
    # The operators are assumed as characters that are non-alphanumeric or _
    # they are not checked against the supported operator for actually being understood at this point
    combined_l = []
    while l:
        if not re.match(OPERATOR_PATTERN, l[0]):
            combined_l.append(l[0])
            del l[0]
        elif re.match(OPERATOR_PATTERN, combined_l[-1]):
            combined_l[-1] = combined_l[-1] + l[0]
            del l[0]
        else:
            combined_l.append(l[0])
            del l[0]
    return combined_l


def parse_equation_to_list(equation: str):
    eq_list = re.split(OPERATOR_PATTERN, equation)
    eq_list = [i.strip() for i in eq_list if i and i != ' ']
    return _combine_consecutive_operators(eq_list)


def _validate_equation_list_formatting(equation: list):
    # first step to a well-defined equations is following this format:
    # - operators sandwiched by params
    # - operators do not exist on peripherals
    # TODO include brackets and order of evaluation
    equation_is_valid = True
    for i, eqn_component in enumerate(equation):
        if i % 2:
            if re.match(EXPRESSION_PATTERN, eqn_component):
                equation_is_valid = False
        else:
            if re.match(OPERATOR_PATTERN, eqn_component):
                equation_is_valid = False
    # check if equation ends on an operator
    if re.match(OPERATOR_PATTERN, equation[-1]):
        equation_is_valid = False

    if not equation_is_valid:
        raise AssertionError(f'Equation {equation} is not valid')
    return equation_is_valid


def _get_operator_hierarchy(equation: list):
    # assumes valid formatting: _validate_equation_list_formatting
    operator_indices = list(range(1, len(equation), 2))
    operator_hierarchy = [OPERATOR_HIERARCHY.index(equation[op_idx]) for op_idx in operator_indices]
    return [op_idx for _, op_idx in sorted(zip(operator_hierarchy, operator_indices))]


def _get_index_of_next_operator(equation: list):
    # assumes valid formatting: _validate_equation_list_formatting
    idx_next = None
    current_hierarchy = len(OPERATOR_HIERARCHY)
    for i in range(1, len(equation), 2):
        op_hchy = OPERATOR_HIERARCHY.index(equation[i])
        if op_hchy <= current_hierarchy:
            idx_next = i
            current_hierarchy = op_hchy
    if idx_next is None:
        raise NotImplemented('')
    return idx_next


def _process_component(component, config):
    def function_template(backend_model, **kwargs):
        return get_param(backend_model, name, tuple([kwargs[var] for var in variables]))

    if re.match(NUMBER_PATTERN, component):
        return float(component)
    elif _is_function(component):
        name, variables = parse_function_string(component)
        return function_template
    elif config['components']:
        return build_equation_from_component(config['components'][component], config)
    else:
        raise NotImplemented(f'Component: {component} could not be processed')



def collapse_equation_list_on_next_operation(equation: list, config: dict):
    # TODO include brackets and order of evaluation
    operator_idx = _get_index_of_next_operator(equation)
    lhs = _process_component(equation[operator_idx - 1], config)
    rhs = _process_component(equation[operator_idx + 1], config)
    equation[operator_idx - 1:operator_idx + 2] = [math_operation(equation[operator_idx], lhs, rhs)]
    return equation


def build_equation_from_string(equation, config):
    # def function_template(backend_model, *args):
    #     build_equation(equation)

    equation = parse_equation_to_list(equation)
    # check equation is valid first:
    _validate_equation_list_formatting(equation)
    # TODO add more logic/domain checks
    while len(equation) > 1:
        equation = collapse_equation_list_on_next_operation(equation, config)
    return equation[0]


def build_equation_from_component(component, config):
    if isinstance(component, list):
        for condition in component:
            condition_succeeded, equation = evaluate_condition(condition)
            if condition_succeeded:
                return build_equation_from_string(equation, config)
    elif isinstance(component, str):
        if _is_function(component):
            # bypass equation build and skip to getting the param
            return _process_component(component, config)
        else:
            # the component is already an equation, does not depend on any conditions
            return build_equation_from_string(component, config)
    else:
        raise NotImplemented(f'Component of type {type(component)} is not understood')


def _is_function(string: str):
    """
    Checks for pattern in string such as: function_name[var1, var2]
    """
    return bool(re.match(FUNCTION_PATTERN, string))


def _extract_function_name(string: str):
    """
    Gives the first word character in the passed string, doesnt check for well-formulation of the function
    """
    return re.search(r'(\w+)', string).groups()[0]


def _extract_function_variables(string: str):
    """
    Gives list of variables from function string
    """
    return string.replace(_extract_function_name(string), '').strip('[]').split(', ')


def parse_function_string(string: str):
    """
    Returns function name and list of variables as a two-tuple, if the function matches the function pattern
    function[var1, var2]
    """
    if _is_function(string):
        return _extract_function_name(string), _extract_function_variables(string)
    else:
        raise NotImplemented(f'String {string} could not be understood as a function')


def evaluate_condition(condition: dict):
    """
    Expects format {'if': string condition, 'then': string equation} or {'else': string equation}
    """
    if 'if' in condition:
        condition_eval = False  # TODO
        equation = condition['then']
    elif 'else' in condition:
        condition_eval = True
        equation = condition['else']
    else:
        raise NotImplemented(f'Given condition {condition} does not match expected "if, then", or "else" format.')

    return condition_eval, equation


def create_valid_constraint_rule(model_data, name, config):
    """
    """

    def function_template(backend_model, **kwargs):
        return build_equation_from_component(config['equation'], backend_model, config)

    # TODO port subset masking here possibly
    # subsets = ...
    # todo? replace function calling on config for conditions within the template
    return function_template
