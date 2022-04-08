"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

constraints.py
~~~~~~~~~~

Create constraints for the model.

"""
import operator
import re

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
OPERATOR_PATTERN = r'([^a-zA-Z0-9_ \,\]\[])'
FUNCTION_PATTERN = r'(\w+)\[\w+(, \w+)?\]'


def math_operation(operator: str, a, b):
    if operator in SUPPORTED_OPERATORS:
        return SUPPORTED_OPERATORS[operator](a, b)
    else:
        raise NotImplemented(f'Operator {operator} is not supported. Supported operators: {list(SUPPORTED_OPERATORS)}')


def _combine_consecutive_operators(l: list):
    # The operators are assumed as characters that are are non-alphanumeric or _
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
    equation_is_valid = True
    for i, eqn_component in enumerate(equation):
        if i % 2:
            if not re.match(OPERATOR_PATTERN, eqn_component):
                equation_is_valid = False
        else:
            if re.match(OPERATOR_PATTERN, eqn_component):
                equation_is_valid = False
    # check if equation ends on an operator
    if re.match(OPERATOR_PATTERN, equation[-1]):
        equation_is_valid = False

    if not equation_is_valid:
        raise AssertionError(f'Equation {equation} is not valid')


def build_equation_from_list(equation: list, config: dict):
    # check equation is valid first:
    _validate_equation_list_formatting(equation)
    # TODO add more logic/domain checks
    # TODO build the eqn
    return equation


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
    def function_template(backend_model, *args):
        for condition in config['equation']:
            condition_succeeded, equation = evaluate_condition(condition)
            if condition_succeeded:
                equation_list = parse_equation_to_list(equation)
                return build_equation_from_list(equation_list, config)

    # TODO port subset masking here possibly
    # subsets = ...
    # todo? replace function calling on config for conditions within the template
    return function_template
