#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "benchmarks_configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init(tiramisu_function_name);
    
    // iterators
    int c00(1024);
    var i00("i00", 0, c00) , i01("i01", 0, c00) ;
    
    // computations
    input input00("input00", {i00, i01}, p_int32);
    input input01("input01", {i00}, p_int32);
    input input02("input02", {i00}, p_int32);

    computation comp03("comp03", {i00, i01}, p_int32);
    computation comp04("comp04", {i00, i01}, p_int32);
    comp03.set_expression(comp03(i00, i01) + input00(i00, i01) * input01(i01));
    comp04.set_expression(comp04(i00, i01) + input00(i01, i00) * input02(i01));

    comp03.then(comp04, i01);
    
    // buffers
    buffer buf03("buf03", {1024}, p_int32, a_output);
    buffer buf04("buf04", {1024}, p_int32, a_output);
    buffer buf00("buf00", {1024, 1024}, p_int32, a_input);
    buffer buf01("buf01", {1024}, p_int32, a_input);
    buffer buf02("buf02", {1024}, p_int32, a_input);

    comp03.store_in(&buf03, {i00});
    comp04.store_in(&buf04, {i00});
    input00.store_in(&buf00);
    input01.store_in(&buf01);
    input02.store_in(&buf02);
    
    std::vector<tiramisu::buffer*> buffs_list = {&buf03, &buf04, &buf00, &buf01, &buf02};
    
    // compile and exit
    if (WHAT_TO_DO == COMPILE)
    {
        comp03.tag_parallel_level(0);
        tiramisu::codegen(buffs_list, exec_obj_filename);
        return 0;
    }
    
    // autoscheduling
    auto_scheduler::simple_generator* gen = new auto_scheduler::simple_generator(max_nb_iters);
    auto_scheduler::evaluate_by_execution* exec_evaluator = 
        new auto_scheduler::evaluate_by_execution(
            tiramisu::global::get_implicit_function(), 
            buffs_list, 
            exec_obj_filename,
            exec_wrapper_cmd
        );
    
    auto_scheduler::search_method* searcher;
    auto_scheduler::evaluator* eval_func;
    auto_scheduler::beam_search_accuracy_evaluator* bs_ae;
    
    if (SEARCH_TYPE == BEAM_SEARCH)
        searcher = new auto_scheduler::beam_search(beam_size, beam_depth, nullptr, gen);
    
    else if (SEARCH_TYPE == BEAM_SEARCH_ACCURACY_EVALUATOR)
    {
        bs_ae = new auto_scheduler::beam_search_accuracy_evaluator(beam_size, beam_depth, nullptr, exec_evaluator, gen);
        searcher = bs_ae;
    }
    
    else if (SEARCH_TYPE == MCTS_SIMPLE)
    {
        searcher = new auto_scheduler::simple_mcts(nb_samples, topk, mcts_depth, nullptr, exec_evaluator, gen);
    }
    
    if (EVALUATOR_TYPE == BY_EXECUTION)
        eval_func = exec_evaluator;
    
    else if (EVALUATOR_TYPE == TREE_LSTM)
        eval_func = new auto_scheduler::tree_lstm_evaluator(py_cmd_path, {py_interface_path});
    
    auto_scheduler::auto_scheduler as(searcher, eval_func);
    as.set_exec_evaluator(exec_evaluator);
    as.find_schedule();
    as.apply_best_schedule();
    
    if (SEARCH_TYPE == BEAM_SEARCH_ACCURACY_EVALUATOR)
        bs_ae->print_evals_list();

    return 0;
}