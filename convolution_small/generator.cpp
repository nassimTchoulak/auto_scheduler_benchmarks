#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "benchmarks_configure.h"

using namespace tiramisu;

int main(int argc, char **argv)
{
    tiramisu::init(tiramisu_function_name);
    
    // iterators
    int c00(8), c01(2), c02(1024), c03(1024), c04(3), c05(3), c06(3), c07(1026), c08(1026);
    var i00("i00", 0, c00) , i01("i01", 0, c01) , i02("i02", 0, c02) , i03("i03", 0, c03) , i04("i04", 0, c04) , i05("i05", 0, c05) , i06("i06", 0, c06) , i07("i07", 0, c07) , i08("i08", 0, c08) ;
    
    // computations
    input input00("input00", {i01}, p_int32);
    input input02("input02", {i00, i04, i07, i08}, p_int32);
    input input03("input03", {i01, i04, i05, i06}, p_int32);

    computation comp01("comp01", {i00, i01, i02, i03}, input00(i01));
    computation comp04("comp04", {i00, i01, i02, i03, i04, i05, i06}, p_int32);
    comp04.set_expression(comp04(i00, i01, i02, i03, i04, i05, i06) + input02(i00, i04, i02 + i05, i03 + i06) * input03(i01, i04, i05, i06));
    
    // buffers
    buffer buf01("buf01", {8, 2, 1024, 1024}, p_int32, a_output);
    buffer buf00("buf00", {2}, p_int32, a_input);
    buffer buf02("buf02", {8, 3, 1026, 1026}, p_int32, a_input);
    buffer buf03("buf03", {2, 3, 3, 3}, p_int32, a_input);

    comp01.store_in(&buf01);
    input00.store_in(&buf00);
    input02.store_in(&buf02);
    input03.store_in(&buf03);
    comp04.store_in(&buf01, {i00, i01, i02, i03});
    
    std::vector<tiramisu::buffer*> buffs_list = {&buf01, &buf00, &buf02, &buf03};
    
    // compile and exit
    if (WHAT_TO_DO == COMPILE)
    {
        comp01.then(comp04, i03);
        
        comp01.tag_parallel_level(0);
        
        /*comp01.interchange(i00, i02);
        comp04.interchange(i00, i02);
        
        comp04.vectorize(i03, 4);
        comp01.vectorize(i03, 4);
        
        comp04.tag_unroll_level(5);
        comp04.tag_unroll_level(6);
        comp04.tag_unroll_level(7);*/
        
        tiramisu::codegen(buffs_list, exec_obj_filename);
        return 0;
    }
    
    comp01.then(comp04, i03);
    
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