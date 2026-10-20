"""
A simplified Gradio demo for Deep Research with basic conversation interface.
This version uses the latest Gradio features with ChatMessage for a modern UI.
"""

import time
import gradio as gr
from gradio import ChatMessage
from loguru import logger
from api.agent.config import get_config
from api.agent.deep_research import (
    deep_research_stream,
    generate_followup_questions,
    process_clarifications,
    write_final_report_stream,
    should_clarify_query
)

# Load configuration
config = get_config()


def run_gradio_demo():
    """Run a modern Gradio demo for Deep Research using ChatMessage"""

    # Conversation state (shared across functions)
    conversation_state = {
        "current_query": "",
        "needs_clarification": False,
        "questions": [],
        "depth": 1,  # 这只是初始默认值，会被UI设置覆盖
        "breadth": 1,  # 这只是初始默认值，会被UI设置覆盖
        "waiting_for_clarification": False,
        "clarification_answers": {},
        "report_mode": True,  # 总是生成详细报告
        "show_details": True,  # 总是显示研究详情
        "last_status": "",  # 跟踪最后一个状态更新
        "search_source": get_config()["research"]["search_source"],
        "history_chat": []
    }

    async def research_with_thinking(message, history, depth, breadth, search_source):
        """
        Process the query with progressive thinking steps shown in the UI
        """
        if not message:
            yield history
            return

        logger.debug(f"Starting research, message: {message}, history: {history}, depth: {depth}, breadth: {breadth}, "
                     f"search_source: {search_source}")
        # 记录用户设置的研究参数
        conversation_state["depth"] = depth
        conversation_state["breadth"] = breadth
        conversation_state["search_source"] = search_source

        # 重置最后状态
        conversation_state["last_status"] = ""

        # 提取历史对话中的用户输入
        history_context = ''
        for msg in history:
            if msg.get("role") == "user":
                q = 'Q:' + msg.get("content", "") + '\n'
                history_context += q
        
        # 记录历史对话到conversation_state
        conversation_state["history_context"] = history_context

        # Check if this is a clarification answer
        if conversation_state["waiting_for_clarification"]:
            async for response in handle_clarification_answer(message, history_context):
                yield response
            return

        # Start with a thinking message - only show the current step
        thinking_msg = ChatMessage(
            content="",
            metadata={"title": "_研究中_", "id": 0, "status": "pending"}
        )

        # 总是添加单独的研究日志消息
        log_msg = ChatMessage(
            content="## 研究过程详情\n\n_实时记录研究步骤和发现_",
            metadata={"title": "_研究日志_", "id": 100}
        )
        yield [thinking_msg, log_msg]

        # 在Gradio界面中告知用户我们正在分析查询
        thinking_msg.content = "分析查询需求中..."
        log_msg.content += "\n\n### 查询处理\n**操作**: 分析查询是否需要澄清\n"
        yield [thinking_msg, log_msg]

        # 需要在Gradio界面中检查是否需要澄清
        needs_clarification = await should_clarify_query(message, history_context)
        if needs_clarification:
            # 需要澄清，生成问题并等待用户回答
            thinking_msg.content = "生成澄清问题..."
            log_msg.content += "\n\n### 查询分析\n**结果**: 需要澄清\n"
            yield [thinking_msg, log_msg]
            
            followup_result = await generate_followup_questions(message, history_context)
            questions = followup_result.get("questions", [])
            
            if questions:
                # 保存问题和状态
                conversation_state["current_query"] = message
                conversation_state["questions"] = questions
                conversation_state["waiting_for_clarification"] = True
                
                # 显示问题给用户
                thinking_msg.content = "请回答以下问题，帮助我更好地理解您的查询:"
                for i, q in enumerate(questions, 1):
                    thinking_msg.content += f"\n{i}. {q.get('question', '')}"
                thinking_msg.metadata["status"] = "pending"
                
                log_msg.content += f"\n\n### 等待用户澄清\n**问题数**: {len(questions)}\n**问题**:\n"
                for i, q in enumerate(questions, 1):
                    log_msg.content += f"{i}. {q.get('question', '')}\n"
                
                yield [thinking_msg, log_msg]
                return  # 等待用户回答
            else:
                # 虽然需要澄清，但没有生成有效问题，继续研究
                thinking_msg.content = "无法生成有效的澄清问题，继续研究..."
                log_msg.content += "\n\n### 查询分析\n**结果**: 需要澄清但无有效问题\n"
                yield [thinking_msg, log_msg]
        else:
            # 不需要澄清，直接继续
            thinking_msg.content = "查询已足够清晰，开始研究..."
            log_msg.content += "\n\n### 查询分析\n**结果**: 查询清晰，无需澄清\n"
            yield [thinking_msg, log_msg]

        # 展示研究配置
        thinking_msg.content = "搜索相关信息中..."
        log_msg.content += f"\n\n### 研究配置\n**搜索提供商**: {search_source}\n**深度**: {depth}\n**广度**: {breadth}\n"
        yield [thinking_msg, log_msg]

        # Track current plan and report for streaming
        report_active = False

        # Perform the research with streaming support - 直接进入研究阶段，不再重复调用should_clarify_query
        async for partial_result in deep_research_stream(
                query=message,
                depth=depth,
                breadth=breadth,
                search_source=search_source,
                history_context=history_context
        ):
            # Update thinking message with current research step
            if partial_result.get("status_update"):
                status = partial_result.get("status_update")
                stage = partial_result.get("stage", "")
                
                # 如果是等待澄清阶段，跳过，因为我们已经在前面处理过了
                if stage == "awaiting_clarification":
                    continue
                    
                # 跳过对查询是否需要澄清的重复处理
                if stage == "analyzing_query" or stage == "clarification_needed":
                    continue

                # 检查状态是否有变化，避免重复的状态更新
                if status != conversation_state["last_status"]:
                    # 记录新状态
                    conversation_state["last_status"] = status

                    # Update the thinking message with current status
                    thinking_msg.content = status

                    # 保存研究计划
                    if stage == "plan_generated" and partial_result.get("research_plan"):
                        research_plan = partial_result.get("research_plan")
                        plan_text = "### 研究计划\n"
                        for i, step in enumerate(research_plan):
                            step_id = step.get("step_id", i + 1)
                            description = step.get("description", "")
                            search_query = step.get("search_query", "")
                            goal = step.get("goal", "")
                            plan_text += f"**步骤 {step_id}**: {description}\n- 查询: {search_query}\n- 目标: {goal}\n\n"

                        log_msg.content += f"\n\n{plan_text}"
                        yield [thinking_msg, log_msg]
                        continue  # 已经更新了UI，跳过下面的通用更新

                    # 更新日志消息，累积显示研究历史
                    timestamp = time.strftime('%H:%M:%S')
                    log_msg.content += f"\n\n### [{timestamp}] {status}\n"

                    # 显示当前研究计划步骤
                    if partial_result.get("current_step"):
                        current_step = partial_result.get("current_step")
                        step_id = current_step.get("step_id", "")
                        description = current_step.get("description", "")
                        log_msg.content += f"\n**当前步骤 {step_id}**: {description}\n"

                    # 显示当前查询
                    if partial_result.get("current_queries"):
                        queries = partial_result.get("current_queries")
                        log_msg.content += "\n**当前并行查询**:\n"
                        for i, q in enumerate(queries, 1):
                            log_msg.content += f"{i}. {q}\n"
                    elif partial_result.get("step_query"):
                        log_msg.content += f"\n**当前查询**: {partial_result.get('step_query')}\n"
                    elif partial_result.get("current_query"):
                        log_msg.content += f"\n**当前查询**: {partial_result.get('current_query')}\n"

                    # Add detailed information based on the stage
                    if stage == "insights_found" and partial_result.get("formatted_new_learnings"):
                        log_msg.content += "\n**新见解**:\n" + "\n".join(
                            partial_result.get("formatted_new_learnings", []))
                        if partial_result.get("formatted_new_urls") and len(
                                partial_result.get("formatted_new_urls")) > 0:
                            log_msg.content += "\n\n**来源**:\n" + "\n".join(
                                partial_result.get("formatted_new_urls", [])[:3])

                    elif stage == "step_completed" and partial_result.get("formatted_step_learnings"):
                        log_msg.content += "\n**步骤总结**:\n" + "\n".join(
                            partial_result.get("formatted_step_learnings", []))

                    elif stage == "analysis_completed" and partial_result.get("formatted_final_findings"):
                        log_msg.content += "\n**主要发现**:\n" + "\n".join(
                            partial_result.get("formatted_final_findings", []))

                        if partial_result.get("gaps"):
                            log_msg.content += "\n\n**研究空白**:\n- " + "\n- ".join(partial_result.get("gaps", []))

                    # Add progress information when available
                    if partial_result.get("progress"):
                        progress = partial_result.get("progress")
                        if "current_step" in progress and "total_steps" in progress:
                            log_msg.content += f"\n\n**进度**: 步骤 {progress['current_step']}/{progress['total_steps']}"
                            if "current_depth" in progress and "max_depth" in progress:
                                log_msg.content += f", 深度 {progress['current_depth']}/{progress['max_depth']}"
                            if "processed_queries" in progress:
                                log_msg.content += f", 已处理 {progress['processed_queries']} 个查询"

                    yield [thinking_msg, log_msg]

            # 从研究结果中直接获取最终报告
            if "final_report" in partial_result and not report_active:
                report_active = True
                current_report = partial_result["final_report"]

                # Create report message
                report_msg = ChatMessage(
                    content=current_report,
                    metadata={"title": "_研究报告_", "id": 2}
                )

                # Complete the thinking message
                thinking_msg.metadata["status"] = "done"

                log_msg.content += "\n\n### 研究完成\n**状态**: 生成了最终报告\n**时间**: " + time.strftime(
                    '%H:%M:%S')
                yield [thinking_msg, log_msg, report_msg]

                # We now have everything, so we can break
                break

        # If we didn't get a final report but finished research, generate the report
        if not report_active:
            # Get the final results
            result = partial_result  # Use the last result from the stream
            learnings = result.get("learnings", [])

            # Show synthesis progress
            thinking_msg.content = "正在整合研究结果和生成报告..."

            log_msg.content += "\n\n### 整合结果\n**状态**: 分析并整合所有收集的信息\n**发现数量**: " + str(
                len(learnings))
            yield [thinking_msg, log_msg]

            # Create report message
            report_msg = ChatMessage(
                content="正在生成详细报告...",
                metadata={"title": "_研究报告_", "id": 2}
            )
            yield [thinking_msg, log_msg, report_msg]

            # Get visited URLs
            visited_urls = result.get("visitedUrls", [])

            # 清除最后状态，避免报告生成状态的重复
            conversation_state["last_status"] = ""

            # 直接生成报告，不需要额外的答案
            report_content = ""
            async for report_chunk in write_final_report_stream(
                    prompt=message,
                    learnings=learnings,
                    visited_urls=visited_urls,
                    history_context=history_context
            ):
                report_content += report_chunk
                report_msg.content = report_content
                yield [thinking_msg, log_msg, report_msg]

            # Complete the thinking message
            thinking_msg.metadata["status"] = "done"

            log_msg.content += "\n\n### 研究结束\n**状态**: 全部完成\n**时间**: " + time.strftime('%H:%M:%S')
            yield [thinking_msg, log_msg, report_msg]

    async def handle_clarification_answer(message, history_context):
        """
        Process the user's answers to clarification questions
        """
        # Reset the waiting flag
        conversation_state["waiting_for_clarification"] = False

        # Get the original query and questions
        query = conversation_state["current_query"]
        questions = conversation_state["questions"]

        # 重置最后状态
        conversation_state["last_status"] = ""

        # Start with a thinking message - only show current step
        thinking_msg = ChatMessage(
            content="解析您的澄清回答...",
            metadata={"title": "_处理中_", "id": 0, "status": "pending"}
        )

        # Always add a separate message for detailed logs
        log_msg = ChatMessage(
            content="## 澄清处理详情\n\n_处理用户回答的澄清问题_",
            metadata={"title": "_研究日志_", "id": 100}
        )
        log_msg.content += f"\n\n### 开始处理\n**原始查询**: {query}\n**用户回答**: {message}\n**时间**: {time.strftime('%H:%M:%S')}\n"
        yield [thinking_msg, log_msg]

        # Simple parsing - assume one answer per line or comma-separated
        lines = [line.strip() for line in message.split('\n') if line.strip()]
        if len(lines) < len(questions):
            # Try comma separation if not enough lines
            if ',' in message:
                lines = [ans.strip() for ans in message.split(',')]

        # Create a dictionary of responses
        user_responses = {}
        for i, q in enumerate(questions):
            key = q.get("key", f"q{i}")
            if i < len(lines) and lines[i]:
                user_responses[key] = lines[i]

        # Process the clarifications
        thinking_msg.content = "处理您的澄清..."

        log_msg.content += f"\n\n### 解析回答\n**解析结果**: 获取了 {len(user_responses)}/{len(questions)} 个回答\n"
        yield [thinking_msg, log_msg]

        # Use await directly with the async function
        clarification_result = await process_clarifications(
            query=query,
            user_responses=user_responses,
            all_questions=questions,
            history_context=history_context
        )

        # Get refined query
        refined_query = clarification_result.get("refined_query", query)

        log_msg.content += f"\n\n### 优化查询\n**原始查询**: {query}\n**优化查询**: {refined_query}\n"
        if clarification_result.get("assumptions"):
            log_msg.content += "**假设**:\n- " + "\n- ".join(clarification_result.get("assumptions"))
        yield [thinking_msg, log_msg]

        # Check if direct answer is available
        if not clarification_result.get("requires_search", True) and clarification_result.get("direct_answer"):
            direct_answer = clarification_result.get("direct_answer", "")

            # Complete the thinking message
            thinking_msg.metadata["status"] = "done"

            # Create answer message
            report_msg = ChatMessage(
                content=direct_answer,
                metadata={"title": "_研究报告_", "id": 2}
            )

            log_msg.content += f"\n\n### 直接回答\n**状态**: 查询可以直接回答，无需搜索\n**时间**: {time.strftime('%H:%M:%S')}\n"
            yield [thinking_msg, log_msg, report_msg]
            return

        # Show research progress
        thinking_msg.content = "基于您的澄清搜索信息..."

        log_msg.content += "\n\n### 开始研究\n**状态**: 需要进行搜索\n**深度**: " + str(
            conversation_state.get("depth", 2)) + "\n**广度**: " + str(conversation_state.get("breadth", 3))
        yield [thinking_msg, log_msg]

        # Track current report for streaming
        report_active = False
        partial_result = {}

        # Perform the research with streaming
        async for partial_result in deep_research_stream(
                query=refined_query,
                depth=conversation_state.get("depth", 1),
                breadth=conversation_state.get("breadth", 1),
                user_clarifications=user_responses,
                search_source=conversation_state.get("search_source", "auto"),
                history_context=history_context
        ):
            # Update thinking message with current research step
            if partial_result.get("status_update"):
                status = partial_result.get("status_update")
                stage = partial_result.get("stage", "")

                # 跳过对查询是否需要澄清的重复处理
                if stage == "analyzing_query" or stage == "clarification_needed" or stage == "awaiting_clarification":
                    continue

                # 检查状态是否有变化，避免重复的状态更新
                if status != conversation_state["last_status"]:
                    # 记录新状态
                    conversation_state["last_status"] = status

                    # Update the thinking message with current status
                    thinking_msg.content = status

                    # 保存研究计划
                    if stage == "plan_generated" and partial_result.get("research_plan"):
                        research_plan = partial_result.get("research_plan")
                        plan_text = "### 研究计划\n"
                        for i, step in enumerate(research_plan):
                            step_id = step.get("step_id", i + 1)
                            description = step.get("description", "")
                            search_query = step.get("search_query", "")
                            goal = step.get("goal", "")
                            plan_text += f"**步骤 {step_id}**: {description}\n- 查询: {search_query}\n- 目标: {goal}\n\n"

                        log_msg.content += f"\n\n{plan_text}"
                        yield [thinking_msg, log_msg]
                        continue  # 已经更新了UI，跳过下面的通用更新

                    # 更新日志消息，累积显示研究历史
                    timestamp = time.strftime('%H:%M:%S')
                    log_msg.content += f"\n\n### [{timestamp}] {status}\n"

                    # 显示当前研究计划步骤
                    if partial_result.get("current_step"):
                        current_step = partial_result.get("current_step")
                        step_id = current_step.get("step_id", "")
                        description = current_step.get("description", "")
                        log_msg.content += f"\n**当前步骤 {step_id}**: {description}\n"

                    # 显示当前查询
                    if partial_result.get("current_queries"):
                        queries = partial_result.get("current_queries")
                        log_msg.content += "\n**当前并行查询**:\n"
                        for i, q in enumerate(queries, 1):
                            log_msg.content += f"{i}. {q}\n"
                    elif partial_result.get("step_query"):
                        log_msg.content += f"\n**当前查询**: {partial_result.get('step_query')}\n"
                    elif partial_result.get("current_query"):
                        log_msg.content += f"\n**当前查询**: {partial_result.get('current_query')}\n"

                    # Add detailed information based on the stage
                    if stage == "insights_found" and partial_result.get("formatted_new_learnings"):
                        log_msg.content += "\n**新见解**:\n" + "\n".join(
                            partial_result.get("formatted_new_learnings", []))
                        if partial_result.get("formatted_new_urls") and len(
                                partial_result.get("formatted_new_urls")) > 0:
                            log_msg.content += "\n\n**来源**:\n" + "\n".join(
                                partial_result.get("formatted_new_urls", [])[:3])

                    elif stage == "step_completed" and partial_result.get("formatted_step_learnings"):
                        log_msg.content += "\n**步骤总结**:\n" + "\n".join(
                            partial_result.get("formatted_step_learnings", []))

                    elif stage == "analysis_completed" and partial_result.get("formatted_final_findings"):
                        log_msg.content += "\n**主要发现**:\n" + "\n".join(
                            partial_result.get("formatted_final_findings", []))

                        if partial_result.get("gaps"):
                            log_msg.content += "\n\n**研究空白**:\n- " + "\n- ".join(partial_result.get("gaps", []))

                    # Add progress information when available
                    if partial_result.get("progress"):
                        progress = partial_result.get("progress")
                        if "current_step" in progress and "total_steps" in progress:
                            log_msg.content += f"\n\n**进度**: 步骤 {progress['current_step']}/{progress['total_steps']}"
                            if "current_depth" in progress and "max_depth" in progress:
                                log_msg.content += f", 深度 {progress['current_depth']}/{progress['max_depth']}"
                            if "processed_queries" in progress:
                                log_msg.content += f", 已处理 {progress['processed_queries']} 个查询"

                    yield [thinking_msg, log_msg]

            # 从研究结果中直接获取最终报告
            if "final_report" in partial_result and not report_active:
                report_active = True
                current_report = partial_result["final_report"]

                # Create report message
                report_msg = ChatMessage(
                    content=current_report,
                    metadata={"title": "_研究报告_", "id": 2}
                )

                # Complete the thinking message
                thinking_msg.metadata["status"] = "done"

                log_msg.content += "\n\n### 研究完成\n**状态**: 生成了最终报告\n**时间**: " + time.strftime(
                    '%H:%M:%S')
                yield [thinking_msg, log_msg, report_msg]

                # We now have everything, so we can break
                break

        # If we didn't get a final report but finished research, generate the report
        if not report_active:
            # Get the final results
            result = partial_result  # Use the last result from the stream
            learnings = result.get("learnings", [])

            # Show synthesis progress
            thinking_msg.content = "正在整合研究结果和生成报告..."

            log_msg.content += "\n\n### 整合结果\n**状态**: 分析并整合所有收集的信息\n**发现数量**: " + str(
                len(learnings))
            yield [thinking_msg, log_msg]

            # Create report message
            report_msg = ChatMessage(
                content="正在生成详细报告...",
                metadata={"title": "_研究报告_", "id": 2}
            )
            yield [thinking_msg, log_msg, report_msg]

            # Get visited URLs
            visited_urls = result.get("visitedUrls", [])

            # 清除最后状态，避免报告生成状态的重复
            conversation_state["last_status"] = ""

            # 直接生成报告，不需要额外的答案
            report_content = ""
            async for report_chunk in write_final_report_stream(
                    prompt=refined_query,
                    learnings=learnings,
                    visited_urls=visited_urls,
                    history_context=history_context
            ):
                report_content += report_chunk
                report_msg.content = report_content
                yield [thinking_msg, log_msg, report_msg]

            # Complete the thinking message
            thinking_msg.metadata["status"] = "done"

            log_msg.content += "\n\n### 研究结束\n**状态**: 全部完成\n**时间**: " + time.strftime('%H:%M:%S')
            yield [thinking_msg, log_msg, report_msg]

    # Create a modern interface using ChatInterface
    demo = gr.ChatInterface(
        fn=research_with_thinking,
        title="🔍 Deep Research",
        description="""使用此工具进行深度研究，我将搜索互联网为您找到回答。Powered by <a href="https://github.com/shibing624/deep-research" target="_blank">Deep Research</a> Made with ❤️ by <a href="https://github.com/shibing624" target="_blank">shibing624</a>""",
        additional_inputs=[
            gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                label="研究深度",
                info="更高 = 更深入但更慢"
            ),
            gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                label="研究广度",
                info="更高 = 更多来源但更慢"
            ),
            gr.Dropdown(
                choices=["auto", "parallel", "serper", "tavily", "mp_search", "qdrant", "none"],
                value=get_config()["research"]["search_source"],
                label="搜索提供商",
                info="要使用的搜索引擎"
            )
        ],
        examples=[
            ["特斯拉股票的最新行情?"],
            ["How does climate change affect biodiversity?"],
            ["中国2024年GDP增长了多少?"],
            ["Explain the differences between supervised and unsupervised machine learning."]
        ],
        type="messages"
    )

    # Launch the demo
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    run_gradio_demo()
