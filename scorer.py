def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def convert_arguments(triggers, entities, roles):
    args = list()
    for trigger_idx, entity_idx, role in roles:
        arg_start, arg_end, _ = entities[entity_idx]
        trigger_label = triggers[trigger_idx][-1]
        args.append((arg_start, arg_end, trigger_label, role))
    return args


def score_graphs(gold_graphs, pred_graphs):
    gold_ent_num = pred_ent_num = ent_match_num = 0
    gold_arg_num = pred_arg_num = arg_idn_num = arg_class_num = 0
    gold_trg_num = pred_trg_num = trg_idn_num = trg_cls_num = 0
    gold_arg_num_nested = pred_arg_num_nested = arg_idn_num_nested = arg_cls_num_nested = 0
    for sen_id, (gold_graph, pred_graph) in enumerate(zip(gold_graphs, pred_graphs)):
        # Entity
        gold_entities = gold_graph.entities
        pred_entities = pred_graph.entities
        gold_ent_num += len(gold_entities)
        pred_ent_num += len(pred_entities)
        ent_match_num += len([entity for entity in pred_entities
                              if entity in gold_entities])

        # Trigger
        gold_triggers = gold_graph.triggers
        pred_triggers = pred_graph.triggers
        pred_triggers_no_repeat = list(set(pred_triggers))
        pred_trg_num += len(pred_triggers_no_repeat)
        for sen_pred_id, (trg_start, trg_end, event_type) in enumerate(pred_triggers_no_repeat):
            matched = [item for sen_gold_id, item in enumerate(gold_triggers)
                       if item[0] == trg_start and item[1] == trg_end]
            if matched:
                trg_idn_num += 1
                if matched[0][-1] == event_type:
                    trg_cls_num += 1
        # remove repeated triggers
        gold_triggers_no_repeat = list(set(gold_triggers))
        gold_trg_num += len(gold_triggers_no_repeat)

        # Argument
        gold_args = convert_arguments(gold_triggers, gold_entities,
                                      gold_graph.roles)
        pred_args = convert_arguments(pred_triggers, pred_entities,
                                      pred_graph.roles)
        pred_arg_num += len(pred_args)
        gold_arg_num += len(gold_args)

        for sen_pred_id, pred_arg in enumerate(pred_args):
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for sen_gold_id, item in enumerate(gold_args)
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1

        # nested part
        gold_nested_args = convert_arguments(gold_triggers, gold_triggers,
                                             gold_graph.nested_roles)
        pred_nested_args = convert_arguments(pred_triggers, pred_triggers,
                                             pred_graph.nested_roles)
        gold_arg_num_nested += len(gold_nested_args)
        pred_arg_num_nested += len(pred_nested_args)
        for pred_arg in pred_nested_args:
            arg_start, arg_end, event_type, role = pred_arg
            gold_idn = {item for item in gold_nested_args
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num_nested += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_cls_num_nested += 1

    ti_p, ti_r, ti_f = compute_f1(pred_trg_num, gold_trg_num, trg_idn_num)
    tc_p, tc_r, tc_f = compute_f1(pred_trg_num, gold_trg_num, trg_cls_num)
    ai_p, ai_r, ai_f = compute_f1(pred_arg_num, gold_arg_num, arg_idn_num)
    ac_p, ac_r, ac_f = compute_f1(pred_arg_num, gold_arg_num, arg_class_num)

    pei_p, pei_r, pei_f = compute_f1(pred_arg_num_nested, gold_arg_num_nested, arg_idn_num_nested)
    pec_p, pec_r, pec_f = compute_f1(pred_arg_num_nested, gold_arg_num_nested, arg_cls_num_nested)

    print('TI: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        ti_p * 100.0, ti_r * 100.0, ti_f * 100.0))
    print('TC: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        tc_p * 100.0, tc_r * 100.0, tc_f * 100.0))
    print('AI: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        ai_p * 100.0, ai_r * 100.0, ai_f * 100.0))
    print('AC: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        ac_p * 100.0, ac_r * 100.0, ac_f * 100.0))
    print('PEI: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        pei_p * 100.0, pei_r * 100.0, pei_f * 100.0))
    print('PEC: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        pec_p * 100.0, pec_r * 100.0, pec_f * 100.0))

    scores = {
        'TI': {'p': ti_p, 'r': ti_r, 'f': ti_f},
        'TC': {'p': tc_p, 'r': tc_r, 'f': tc_f},
        'AI': {'p': ai_p, 'r': ai_r, 'f': ai_f},
        'AC': {'p': ac_p, 'r': ac_r, 'f': ac_f},
        'PEI': {'p': pei_p, 'r': pei_r, 'f': pei_f},
        'PEC': {'p': pec_p, 'r': pec_r, 'f': pec_f},

    }
    return scores
