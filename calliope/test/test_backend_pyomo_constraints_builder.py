import pytest
import calliope.backend.constraints as constraints
from pyomo.core import ConcreteModel
import calliope.backend.pyomo.util


def test_parsing_basic_equation():
    assert constraints.parse_equation_to_list('some_word < another_word') == ['some_word', '<', 'another_word']


def test_parsing_equation_with_complex_operation():
    assert constraints.parse_equation_to_list('some_word <= another_word') == ['some_word', '<=', 'another_word']


def test_parsing_equation_with_square_brackets():
    assert constraints.parse_equation_to_list(
        'some_word[x, y] <= another_word[x, y]') == ['some_word[x, y]', '<=', 'another_word[x, y]']


def test_parsing_equation_with_round_brackets():
    assert constraints.parse_equation_to_list(
        'A <= (B + C) * D') == ['A', '<=', ['B', '+', 'C'], '*', 'D']


def test_parsing_equation_with_multiple_round_brackets():
    assert constraints.parse_equation_to_list(
        'A <= (B * (C - D)) * E') == ['A', '<=', ['B', '*', ['C', '-', 'D']], '*', 'E']


def test_valid_simple_equation_passes_formatting_validation():
    constraints._validate_equation_list_formatting(['some_word', '<=', 'another_word'])


def test_valid_complex_equation_passes_formatting_validation():
    constraints._validate_equation_list_formatting(['some_word', '<=', 'another_word', '<=', 'something'])


def test_equation_with_square_brackets_passes_formatting_validation():
    assert constraints._validate_equation_list_formatting(['some_word[x, y]', '<=', 'another_word[x, y]'])


def test_equation_without_rhs_raises_error_when_validating_formatting():
    with pytest.raises(AssertionError) as error_info:
        constraints._validate_equation_list_formatting(['<=', 'some_word'])
    assert "not valid" in str(error_info.value)


def test_equation_without_lhs_raises_error_when_validating_formatting():
    with pytest.raises(AssertionError) as error_info:
        constraints._validate_equation_list_formatting(['some_word', '<='])
    assert "not valid" in str(error_info.value)


def test_equation_with_multiple_consecutive_operators_raises_error_when_validating_formatting():
    with pytest.raises(AssertionError) as error_info:
        constraints._validate_equation_list_formatting(['some_word', '<=', '=', 'blah'])
    assert "not valid" in str(error_info.value)


def test_string_function_with_single_var_is_indeed_a_function():
    assert constraints._is_function('resource_unit[node]')


def test_string_function_with_multiple_vars_is_indeed_a_function():
    assert constraints._is_function('resource_unit[node, tech]')


def test_string_function_with_malformed_vars_is_not_a_function():
    assert not constraints._is_function('resource_unit[node, ]')


def test_string_function_with_malformed_brackets_is_not_a_function():
    assert not constraints._is_function('resource_unit[node, tech')


def test_sting_equation_is_not_a_function():
    assert not constraints._is_function('resource_unit[node, tech] + sth[node, tech]')


def test_extracting_function_name_from_well_formed_function():
    assert constraints._extract_function_name('resource_unit[node, tech]') == 'resource_unit'


def test_extracting_function_name_from_malformed_function_also_works():
    assert constraints._extract_function_name('resource_unit[node, tec') == 'resource_unit'


def test_extracting_function_variables_from_well_formed_function():
    assert constraints._extract_function_variables('resource_unit[node, tech]') == ['node', 'tech']


def test_extracting_function_variables_with_single_var_function():
    assert constraints._extract_function_variables('resource_unit[node]') == ['node']


def test_extracting_function_variables_from_malformed_function_also_works():
    assert constraints._extract_function_variables('resource_unit[node, tech') == ['node', 'tech']


def test_ordering_operators_in_simple_equation():
    idx = constraints._get_index_of_next_operator(['A', '==', 'B', '+', 'C'])
    assert idx == 3


def test_ordering_operators_in_a_complex_equation():
    idx = constraints._get_index_of_next_operator(['A', '==', 'B', '+', 'C', '*', 'D'])
    assert idx == 5


def test_collapse_simple_equation_once():
    collapsed_list = constraints.collapse_equation_list_on_next_operation(['3', '==', '2', '+', '1'], config={})
    assert len(collapsed_list) == 3
    operation = collapsed_list[-1]
    assert callable(operation)
    assert operation(backend_model=None) == 3
    assert collapsed_list == ['3', '==', operation]


def test_collapse_complex_equation_once():
    collapsed_list = constraints.collapse_equation_list_on_next_operation(
        ['11', '==', '3', '+', '2', '*', '4'], config={})
    assert len(collapsed_list) == 5
    operation = collapsed_list[-1]
    assert callable(operation)
    assert operation(backend_model=None) == 8
    assert collapsed_list == ['11', '==', '3', '+', operation]


def test_collapse_equation_with_brackets_once_collapses_bracket_inside_and_the_next_operation():
    collapsed_list = constraints.collapse_equation_list_on_next_operation(
        ['20', '==', ['3', '+', '2'], '*', '4'], config={})
    assert len(collapsed_list) == 3
    operation = collapsed_list[-1]
    assert callable(operation)
    assert operation(backend_model=None) == 20
    assert collapsed_list == ['20', '==', operation]


def test_collapse_equation_with_multiple_brackets_collapses_brackets_inside_and_the_next_operation():
    collapsed_list = constraints.collapse_equation_list_on_next_operation(
        ['184', '==', ['6', '+', ['2', '**', '3'], '*', '5'], '*', '4'], config={})
    # (6 + (8 * 5)) * 4
    assert len(collapsed_list) == 3
    operation = collapsed_list[-1]
    assert callable(operation)
    assert operation(backend_model=None) == 184
    assert collapsed_list == ['184', '==', operation]


def test_building_wholly_numerical_equation():
    eqn = constraints.build_equation_from_string(
        '1 + 2',
        config={}
    )
    assert eqn(backend_model=None) == 3


def test_wholly_numerical_equation_remains_constant_with_kwargs():
    eqn = constraints.build_equation_from_string(
        '1 + 2',
        config={}
    )
    assert eqn(backend_model=None, x=1, y=1) == 3


@pytest.fixture()
def two_param_backend_model():
    backend_model = ConcreteModel()
    backend_model.A = {(1, 2): 3}
    backend_model.B = {(1, 2): 7}
    return {'backend_model': backend_model,
            'param_1': 'A[x, y]', 'param_2': 'B[x, y]',
            'x_idx': 1, 'y_idx': 2,
            'param_1_value': 3, 'param_2_value': 7}


def test_building_simple_equation(two_param_backend_model):
    eqn = constraints.build_equation_from_component(
        f'{two_param_backend_model["param_1"]} + {two_param_backend_model["param_2"]}',
        config={}
    )
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == two_param_backend_model['param_1_value'] + two_param_backend_model['param_2_value']


def test_building_equation_with_multiple_operations(two_param_backend_model):
    eqn = constraints.build_equation_from_component(
        f'{two_param_backend_model["param_1"]} + {two_param_backend_model["param_2"]} > 0',
        config={}
    )
    assert (two_param_backend_model['param_1_value'] + two_param_backend_model['param_2_value']) > 0
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == True


def test_equation_with_conditional_statement_executes_if_then_component(two_param_backend_model):
    config = {
        'equation': [
            {'if': f'{two_param_backend_model["param_1"]} == {two_param_backend_model["param_1_value"]}', # this is true
             'then': f'{two_param_backend_model["param_1"]} == {two_param_backend_model["param_2"]}'}, # expected to evaluate to false
            {'else': f'{two_param_backend_model["param_1"]} < {two_param_backend_model["param_2"]}'}  # expected to evaluates to true
        ]
    }
    eqn = constraints.build_equation_from_component(
        config['equation'],
        config=config
    )
    assert (two_param_backend_model['param_1_value'] == two_param_backend_model['param_2_value']) == False
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == False


def test_equation_with_conditional_statement_executes_else_component(two_param_backend_model):
    config = {
        'equation': [
            {'if': f'{two_param_backend_model["param_1"]} != {two_param_backend_model["param_1_value"]}', # this is true
             'then': f'{two_param_backend_model["param_1"]} == {two_param_backend_model["param_2"]}'}, # expected to evaluate to false
            {'else': f'{two_param_backend_model["param_1"]} < {two_param_backend_model["param_2"]}'}  # expected to evaluates to true
        ]
    }
    eqn = constraints.build_equation_from_component(
        config['equation'],
        config=config
    )
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == (two_param_backend_model['param_1_value'] < two_param_backend_model['param_2_value'])


def test_building_equation_with_components(two_param_backend_model):
    eqn = constraints.build_equation_from_component(
        'comp_a + comp_b',
        config={'components': {'comp_a': two_param_backend_model["param_1"], 'comp_b': two_param_backend_model["param_2"]}}
    )
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == two_param_backend_model['param_1_value'] + two_param_backend_model['param_2_value']


@pytest.mark.xfail()
def test_building_equation_with_conditional_components(two_param_backend_model):
    eqn = constraints.build_equation_from_component(
        'comp_a + comp_b',
        config={'components': {'comp_a': two_param_backend_model["param_1"], 'comp_b': two_param_backend_model["param_2"]}}
    )
    assert eqn(
        backend_model=two_param_backend_model['backend_model'],
        x=two_param_backend_model['x_idx'],
        y=two_param_backend_model['y_idx']
    ) == two_param_backend_model['param_1_value'] + two_param_backend_model['param_2_value']


@pytest.fixture()
def balance_supply_constraint_config():
    return {
        'components': {
            'available_resource': [
                {'if': "resource_unit[node, tech] == 'energy_per_area'",
                 'then': 'resource[node, tech, timestep] * resource_scale[node, tech] * resource_area[node, tech]'},
                {'if': "resource_unit[node, tech] == 'energy_per_cap'",
                 'then': 'resource[node, tech, timestep] * resource_scale[node, tech] * energy_cap[node, tech]'},
                {'else': 'resource[node, tech, timestep] * resource_scale[node, tech]'}
            ],
            'carrier_prod_div_energy_eff': 'carrier_prod[carrier, node, tech, timestep] / energy_eff[node, tech, timestep]'
        },
        'equation': [
            {'if': 'energy_eff[node, tech, timestep] == 0', 'then': 'carrier_prod[carrier, node, tech, timestep] == 0'},
            {'if': 'force_resource[node, tech] == True', 'then': 'carrier_prod_div_energy_eff == available_resource'},
            {'if': 'resource_min_use[node, tech]',
             'then': 'resource_min_use[node, tech] * available_resource <= carrier_prod_div_energy_eff <= available_resource'},
            {'else': 'carrier_prod_div_energy_eff <= available_resource'}
        ],
        'foreach': ['carrier in carriers', 'node in nodes', 'tech in techs', 'timestep in timesteps'],
        'where': ['resource', 'and', 'inheritance(supply)']
    }


@pytest.fixture()
def balance_supply_plus_constraint_config():
    return {
        'components': {
            'available_resource': 'resource_con[node, tech, timestep] * resource_eff[node, tech, timestep]',
            'carrier_prod_incl_losses': [
                {'if': 'energy_eff[node, tech, timestep] == 0 | parasitic_eff[node, tech, timestep] == 0',
                 'then': 0},
                {
                    'else': 'carrier_prod[carrier, node, tech, timestep] / (energy_eff[node, tech, timestep] * parasitic_eff[node, tech, timestep])'}
            ],
            'storage_at_timestep_start': [
                {'if`': 'include_storage[node, tech] == True & run.cyclic_storage == False & get_index{timestep}=0',
                 'then': 'storage_initial[node, tech] * storage_cap[node, tech]'},
                {
                    'if': 'include_storage[node, tech] == True & exists{storage_inter_cluster} & lookup_cluster_first_timestep[timestep] == True',
                    'then': 0},
                {
                    'else': '((1 - storage_loss[node, tech]) ** timestep_resolution[previous_step]) * storage[node, tech, previous_step]'}
            ]
        },
        'equation': [
            {'if': 'include_storage[node, tech] == False', 'then': 'available_resource == carrier_prod_incl_losses'},
            {
                'else': 'storage[node, tech, timestep] == storage_at_timestep_start + available_resource - carrier_prod_incl_losses'}],
        'foreach': ['carrier in carriers', 'node in nodes', 'tech in techs', 'timestep in timesteps'],
        'index_items': {
            'previous_step': {
                'on_dimension': 'timesteps', 'select': [
                    {'if': 'include_storage[node, tech] == True & run.cyclic_storage == True & get_index{timesteps}==0',
                     'then': 'get_item{timesteps, -1}'},
                    {'if': 'include_storage[node, tech] == True & get_index{timesteps}>0',
                     'then': 'get_item{timesteps, get_index{timestep} - 1}'},
                    {
                        'if': 'include_storage[node, tech] == True & get_index{timesteps}>0 & clusters & lookup_cluster_first_timestep[timestep] == True',
                        'then': 'lookup_cluster_last_timestep[timestep]'}
                ]
            }
        },
        'where': ['inheritance(supply_plus)']
    }


@pytest.mark.xfail()
def test_creating_constraint_function_from_constraint_dictionary(balance_supply_constraint_config):
    rule = constraints.create_valid_constraint_rule(None, 'balance_supply', balance_supply_constraint_config)
    rule(backend_model=None)


@pytest.mark.xfail()
def test_creating_multiple_constraint_functions_results_in_unique_functions(balance_supply_constraint_config,
                                                                            balance_supply_plus_constraint_config):
    rule_bs = constraints.create_valid_constraint_rule(None, 'balance_supply', balance_supply_constraint_config)
    rule_bss = constraints.create_valid_constraint_rule(None, 'balance_supply_plus',
                                                        balance_supply_plus_constraint_config)

    assert rule_bs != rule_bss
    assert rule_bs(None) != rule_bss(None)
