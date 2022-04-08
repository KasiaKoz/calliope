import pytest
import calliope.backend.constraints as constraints


def test_parsing_basic_equation():
    assert constraints.parse_equation_to_list('some_word < another_word') == ['some_word', '<', 'another_word']


def test_parsing_equation_with_complex_operation():
    assert constraints.parse_equation_to_list('some_word <= another_word') == ['some_word', '<=', 'another_word']


def test_parsing_equation_with_brackets():
    assert constraints.parse_equation_to_list(
        'some_word[x, y] <= another_word[x, y]') == ['some_word[x, y]', '<=',  'another_word[x, y]']


def test_validating_formatting_of_valid_simple_equation_passes():
    constraints._validate_equation_list_formatting(['some_word', '<=', 'another_word'])


def test_validating_formatting_of_valid_complex_equation_passes():
    constraints._validate_equation_list_formatting(['some_word', '<=', 'another_word', '<=', 'something'])


def test_validating_formatting_of_equation_without_rhs_raises_error():
    with pytest.raises(AssertionError) as error_info:
        constraints._validate_equation_list_formatting(['<=', 'some_word'])
    assert "not valid" in str(error_info.value)


def test_validating_formatting_of_equation_without_lhs_raises_error():
    with pytest.raises(AssertionError) as error_info:
        constraints._validate_equation_list_formatting(['some_word', '<='])
    assert "not valid" in str(error_info.value)


def test_validating_formatting_of_equation_with_multiple_consecutive_operators_raises_error():
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


def test_creating_constraint_function_from_constraint_dictionary(balance_supply_constraint_config):
    rule = constraints.create_valid_constraint_rule(None, 'balance_supply', balance_supply_constraint_config)
    rule(None)


def test_creating_multiple_constraint_functions_results_in_unique_functions(balance_supply_constraint_config,
                                                                            balance_supply_plus_constraint_config):
    rule_bs = constraints.create_valid_constraint_rule(None, 'balance_supply', balance_supply_constraint_config)
    rule_bss = constraints.create_valid_constraint_rule(None, 'balance_supply_plus',
                                                        balance_supply_plus_constraint_config)

    assert rule_bs != rule_bss
    assert rule_bs(None) != rule_bss(None)
