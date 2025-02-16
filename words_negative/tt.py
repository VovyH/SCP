def process_input_sentences(self, harmful_behaviors):
    """
    对输入句子列表逐条进行处理，生成善意句子。
    动态调整 reference_token，初始为 "avoid"。
    如果失败则选择近义词重试，直到成功为止。
    """
    results = []
    positive_sentences = []

    try:
        # 获取已处理的行数
        start_index = self.get_processed_line_count()
        if start_index == 0:
            print("No processed data found or file is empty. Starting from scratch.")
        
        # 使用tqdm包装循环，显示进度条
        for origin_input in tqdm(enumerate(harmful_behaviors[start_index:], start=start_index), desc="Processing sentences"):
            success = False
            current_token = self.reference_token  # 当前尝试的 reference_token

            while not success:
                try:
                    # Step 1: 原输入分词并标注词性
                    original_tags = split_and_tag(origin_input)

                    # Step 2: 使用当前 reference_token 生成善意句子
                    positive_sentence = generate_positive_sentence(
                        config_args.test_model, original_tags, current_token
                    )

                    # Step 3: 善意句子分词并标注词性
                    positive_tags = split_and_tag(positive_sentence)

                    # Step 4: 对比原句和善意句，找到多余的 Token
                    additional_tokens = compare_tokens(original_tags, positive_tags)

                    # Step 5: 防御机制只能选择一个 Token，且不能是新增的非谓语动词
                    defense_mechanism = None
                    for token in additional_tokens:
                        # 检查 token 是否是新增非谓语动词
                        token_tag = [tag for tok, tag in positive_tags if tok == token]
                        if token_tag and token_tag[0].startswith("VB") and token_tag[0] != "VB":
                            continue  # 排除非谓语动词
                        defense_mechanism = token  # 选择第一个符合条件的 token
                        break

                    # Step 6: 对善意句子进行安全性评估
                    is_safe = evaluate_sentence_safety(config_args.test_model, positive_sentence)

                    if is_safe:
                        # 成功生成，标记为成功
                        success = True
                        defense_mechanism = current_token
                        self.token_pool[current_token]["success"] += 1  # 成功的 token 分子加 1
                    else:
                        # Step 7: 如果失败，选择当前 reference_token 的近义词重试
                        synonyms = get_synonyms(current_token)
                        if synonyms:
                            # 选择概率值最高的近义词作为新的 reference_token
                            current_token = max(synonyms, key=lambda x: self.calculate_probability(x))
                        else:
                            # 如果没有近义词，回退到默认的 "avoid"
                            current_token = "avoid"

                    # 更新 token 的总使用次数
                    self.token_pool[current_token]["total"] += 1

                except Exception as e:
                    # 出现错误时保存当前结果并输出错误
                    print(f"An error occurred while processing: {e}")
                    self.save_results(results, positive_sentences)
                    raise e  # 重新抛出异常

            # Step 8: 保存结果
            positive_sentences.append(positive_sentence)
            results.append(
                {
                    "original_sentence": origin_input,
                    "positive_sentence": positive_sentence,
                    "defense_mechanism": defense_mechanism,  # 保存单一的防御机制
                    "is_safe": is_safe,
                    "reference_token": current_token,
                    "reference_token_probability": self.calculate_probability(current_token),  # 添加当前reference_token的概率
                    "test_model": config_args.test_model  # 添加当前测试模型
                }
            )

            # Step 9: 更新 reference_token，选择概率值最高的 token
            self.reference_token = max(self.token_pool, key=lambda x: self.calculate_probability(x))

    except Exception as e:
        print(f"An error occurred in processing input sentences: {e}")

    return results, positive_sentences
