"""
Deep research functionality for comprehensive multi-step research.
"""

import asyncio
import json
import traceback
import inspect
from typing import Optional, Callable, Dict, List, Any, Union, Tuple, AsyncGenerator

import loguru
from loguru import logger

from api.agent.config import get_config
from api.agent.providers import get_search_provider
from api.agent.search_context import SearchContext
from api.agent.utils import format_source_metadata, add_event_loop_policy
from api.agent.prompts import (
    SYSTEM_PROMPT,
    SHOULD_CLARIFY_QUERY_PROMPT,
    FOLLOW_UP_QUESTIONS_PROMPT,
    PROCESS_CLARIFICATIONS_PROMPT,
    RESEARCH_PLAN_PROMPT,
    SUMMARIZE_SEARCH_RESULTS_PROMPT,
    RESEARCH_FROM_CONTENT_PROMPT,
    RESEARCH_SUMMARY_PROMPT,
    FINAL_REPORT_PROMPT,
    FINAL_ANSWER_PROMPT
)
from api.agent.model_utils import generate_completion, generate_json_completion
from api.agent.search_utils import search_and_summarize, concurrent_search


async def should_clarify_query(query: str, history_context: str = '') -> bool:
    """
    Use the language model to determine if a query needs clarification.
    """
    try:
        prompt = SHOULD_CLARIFY_QUERY_PROMPT.format(query=query, history_context=history_context)
        result = await generate_completion(prompt, temperature=0)
        logger.debug(f"query: {query}, should clarify query result: {result}")

        # 检查结果是否包含肯定回答（yes/y）
        needs_clarification = "yes" in result.lower() or 'y' in result.lower()
        return needs_clarification
    except Exception as e:
        logger.error(f"error: {str(e)}")
        return True


async def generate_followup_questions(query: str, history_context: str = '') -> Dict[str, Any]:
    """
    Generate clarifying follow-up questions for the given query.
    
    Args:
        query: The user's research query
        
    Returns:
        Dict containing whether clarification is needed and questions
    """
    try:
        # Format the prompt
        prompt = FOLLOW_UP_QUESTIONS_PROMPT.format(query=query, history_context=history_context)

        # Generate the followup questions
        result = await generate_json_completion(prompt, system_message=SYSTEM_PROMPT, temperature=0.7)

        # Ensure the expected structure
        if "needs_clarification" not in result:
            result["needs_clarification"] = False

        if "questions" not in result:
            result["questions"] = []

        logger.debug(f"Follow-up questions: {result}")
        return result

    except Exception as e:
        logger.error(f"Error generating followup questions: {str(e)}")
        return {
            "needs_clarification": False,
            "questions": []
        }


async def process_clarifications(
        query: str,
        user_responses: Dict[str, str],
        all_questions: List[Dict[str, Any]],
        history_context: str = ''
) -> Dict[str, Any]:
    """
    Process user responses to clarification questions and refine the query.
    
    Args:
        query: The original query
        user_responses: Dict mapping question keys to user responses
        all_questions: List of all questions that were asked
        
    Returns:
        Dict with refined query and other information
    """
    try:
        # Format the questions and responses for the prompt
        clarifications = []
        unanswered = []

        for question in all_questions:
            key = question.get("key", "")
            question_text = question.get("question", "")
            default = question.get("default", "")

            if key in user_responses and user_responses[key]:
                clarifications.append(f"Q: {question_text}\nA: {user_responses[key]}")
            else:
                clarifications.append(f"Q: {question_text}\nA: [Not answered by user]")
                unanswered.append(f"Q: {question_text}\nDefault: {default}")

        # Check if all questions were unanswered - if so, modify the prompt
        all_unanswered = len(unanswered) == len(all_questions) and len(all_questions) > 0

        # Format the prompt
        if all_unanswered:
            prompt = f"""
I'm reviewing a user query where they chose not to provide any clarifications.

Chat history: {history_context}

Original query: {query}

The user was asked the following clarification questions but chose not to answer any:
{" ".join(unanswered)}

Since the user didn't provide any clarifications, please:
1. Analyze the original query as comprehensively as possible
2. Make reasonable assumptions for all ambiguous aspects
3. Determine if this is a simple factual query that doesn't require search
4. If possible, provide a direct answer along with the refined query

Format your response as a valid JSON object with the following structure:
{{
  "refined_query": "The refined query with all possible considerations",
  "assumptions": ["List of all assumptions made"],
  "requires_search": true/false (boolean indicating if this query needs web search or can be answered directly),
  "direct_answer": "If requires_search is false, provide a comprehensive direct answer here, otherwise empty string"
}}

Since the user chose not to provide clarifications, be as thorough and comprehensive as possible in your analysis and answer.
"""
        else:
            prompt = PROCESS_CLARIFICATIONS_PROMPT.format(
                query=query,
                clarifications="\n\n".join(clarifications),
                unanswered_questions="\n\n".join(unanswered) if unanswered else "None",
                history_context=history_context
            )

        # Generate the clarifications processing
        result = await generate_json_completion(prompt, temperature=0.7)

        # Ensure the expected structure
        if "refined_query" not in result:
            result["refined_query"] = query

        if "assumptions" not in result:
            result["assumptions"] = []

        if "requires_search" not in result:
            result["requires_search"] = True

        if "direct_answer" not in result:
            result["direct_answer"] = ""

        logger.debug(f"Processed clarifications: {result}")
        return result

    except Exception as e:
        logger.error(f"Error processing clarifications: {str(e)}")
        return {
            "refined_query": query,
            "assumptions": [],
            "requires_search": True,
            "direct_answer": ""
        }


async def generate_research_plan(query: str, history_context: str = "") -> Dict[str, Any]:
    """
    Generate a research plan with a variable number of steps based on the query complexity.
    
    Args:
        query: The research query
        
    Returns:
        Dict containing the research plan with steps
    """
    try:
        # Format the prompt
        prompt = RESEARCH_PLAN_PROMPT.format(query=query, history_context=history_context)

        # Generate the research plan
        result = await generate_json_completion(prompt, system_message=SYSTEM_PROMPT, temperature=0.7)

        # Ensure the expected structure
        if "steps" not in result or not result["steps"]:
            # Create a default single-step plan if none was provided
            result["steps"] = [{
                "step_id": 1,
                "description": "Research the query",
                "search_query": query,
                "goal": "Find information about the query"
            }]

        if "assessments" not in result:
            result["assessments"] = "No complexity assessment provided"

        logger.debug(f"Research plan: {result}")
        return result

    except Exception as e:
        logger.error(f"Error generating research plan: {str(e)}")
        # Return a simple default plan
        return {
            "assessments": "Error occurred, using default plan",
            "steps": [{
                "step_id": 1,
                "description": "Research the query",
                "search_query": query,
                "goal": "Find information about the query"
            }]
        }


async def summarize_search_results(query: str, search_results: str) -> str:
    """
    Summarize search results for a query.
    
    Args:
        query: The search query
        search_results: Formatted search results text
        
    Returns:
        Summarized search results
    """
    try:
        # Format the prompt
        prompt = SUMMARIZE_SEARCH_RESULTS_PROMPT.format(
            query=query,
            search_results=search_results
        )

        # Generate the summary
        summary = await generate_completion(prompt, temperature=0.7)
        return summary

    except Exception as e:
        logger.error(f"Error summarizing search results: {str(e)}")
        return f"Error summarizing results for '{query}': {str(e)}"


async def research_step(
        query: str,
        depth: int,
        breadth: int,
        config: Dict[str, Any],
        on_progress: Optional[Callable] = None,
        search_provider=None,
        step_info: Dict[str, Any] = None,
        all_learnings: List[str] = None
) -> Dict[str, Any]:
    """
    Perform a single step of the research process.
    
    Args:
        query: The query to research
        depth: Current depth level
        breadth: Number of parallel queries to run
        config: Configuration dictionary
        on_progress: Optional callback for progress updates
        search_provider: Search provider instance
        step_info: Information about the current research plan step
        all_learnings: List of learnings accumulated so far
        
    Returns:
        Dict with research results
    """
    if all_learnings is None:
        all_learnings = []

    if search_provider is None:
        search_provider = get_search_provider()

    # Progress update
    if on_progress:
        progress_data = {
            "currentDepth": depth,
            "totalDepth": config["research"]["default_depth"],
            "completedQueries": 0,
            "totalQueries": breadth,
            "currentQuery": query
        }

        # Check if on_progress is a coroutine function
        if inspect.iscoroutinefunction(on_progress):
            await on_progress(progress_data)
        else:
            on_progress(progress_data)

    # Get the search results
    loguru.logger.debug(query)
    search_result = await search_and_summarize(query, config, search_provider)
    loguru.logger.debug(search_result)

    search_results_text = search_result["summary"]

    # Summarize the search results
    summary = await summarize_search_results(query, search_results_text)

    # If we've reached the maximum depth, just return the summary
    if depth <= 0:
        return {
            "summary": summary,
            "urls": search_result["urls"],
            "nextQueries": [],
            "learnings": []
        }

    # Figure out what to search next based on learnings so far
    prompt = RESEARCH_FROM_CONTENT_PROMPT.format(
        query=query,
        current_step=json.dumps(step_info, ensure_ascii=False) if step_info else "No specific step information",
        content=summary,
        next_queries_count=breadth
    )

    # Generate the next queries and learnings
    research_output = await generate_json_completion(prompt, temperature=0.7)

    # Extract the next queries and learnings
    next_queries = research_output.get("nextQueries", [])
    new_learnings = research_output.get("learnings", [])

    # Update learnings
    all_learnings.extend(new_learnings)

    return {
        "summary": summary,
        "urls": search_result["urls"],
        "nextQueries": next_queries,
        "learnings": new_learnings
    }


async def deep_research(
        query: str,
        breadth: int = None,
        depth: int = None,
        on_progress: Optional[Callable] = None,
        user_clarifications: Dict[str, str] = None,
        search_source: Optional[str] = None,
        history_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform deep research by iteratively searching based on learned information.
    
    Args:
        query: The research query
        breadth: Number of parallel queries to run at each step
        depth: Number of iterations to perform
        on_progress: Optional callback function for progress updates
        user_clarifications: User responses to clarification questions
        search_source: Optional search provider to use
        
    Returns:
        Dict with research results including learnings and visited URLs
    """
    # Load configuration
    config = get_config()

    # Use defaults from config if not specified
    if breadth is None:
        breadth = config["research"]["default_breadth"]
    if depth is None:
        depth = config["research"]["default_depth"]

    # Initialize tracking variables
    visited_urls = []
    all_learnings = []

    # Initialize search provider
    search_provider = get_search_provider(search_source=search_source, context=SearchContext())

    # Helper function to handle progress callbacks
    async def update_progress(progress_data):
        if on_progress:
            if inspect.iscoroutinefunction(on_progress):
                await on_progress(progress_data)
            else:
                on_progress(progress_data)

    try:
        # Step 1: 先判断是否需要生成澄清问题
        await update_progress({"step": "query_analysis", "message": "Analyzing query for clarity"})
        needs_clarification = False
        questions = []


        # 不需要澄清，直接使用原始查询
        refined_query = query
        await update_progress({"step": "clarification_skipped", "message": "Query is clear, skipping clarification"})

        # Step 3: Generate research plan with variable steps
        await update_progress({"step": "planning", "message": "Generating research plan"})
        logger.debug(refined_query)
        plan_result = await generate_research_plan(refined_query, history_context)
        logger.debug(plan_result)

        steps = plan_result.get("steps", [])

        # Track step summaries for the final analysis
        step_summaries = []

        # Iterate through each step in the research plan
        for step_idx, step in enumerate(steps):
            step_id = step.get("step_id", step_idx + 1)
            description = step.get("description", f"Research step {step_id}")
            search_query = step.get("search_query", refined_query)

            await update_progress({
                "currentDepth": depth,
                "totalDepth": depth,
                "completedQueries": step_idx,
                "totalQueries": len(steps),
                "currentQuery": f"Step {step_id}: {description}"
            })

            # Create a queue of queries to process for this step
            queries_to_process = [search_query]
            step_urls = []
            step_learnings = []

            # Process the initial query and follow-ups
            current_depth = depth
            total_processed = 0

            while queries_to_process and current_depth > 0:
                # Get the next queries to process (up to breadth)
                current_queries = queries_to_process[:breadth]
                queries_to_process = queries_to_process[breadth:]

                # Research these queries concurrently
                await update_progress({
                    "currentDepth": current_depth,
                    "totalDepth": depth,
                    "completedQueries": total_processed,
                    "totalQueries": len(queries_to_process) + len(current_queries),
                    "currentQuery": f"Processing {len(current_queries)} queries concurrently"
                })

                # Process each query in the current batch
                research_tasks = []
                for current_query in current_queries:
                    task = research_step(
                        query=current_query,
                        depth=current_depth,
                        breadth=breadth,
                        config=config,
                        on_progress=on_progress,
                        search_provider=search_provider,
                        step_info=step,
                        all_learnings=all_learnings
                    )
                    research_tasks.append(task)

                # Execute tasks with concurrency
                results = await asyncio.gather(*research_tasks)

                # Process the results
                for result in results:
                    # Update tracking variables
                    step_urls.extend(result["urls"])
                    step_learnings.extend(result["learnings"])
                    total_processed += 1

                    # Add the new queries to our queue
                    next_queries = result["nextQueries"]
                    queries_to_process.extend(next_queries)

                # Decrement depth for next iteration
                current_depth -= 1

            # Update visited URLs and learnings
            visited_urls.extend(step_urls)
            all_learnings.extend(step_learnings)

            # Save step summary
            step_summaries.append({
                "step_id": step_id,
                "description": description,
                "learnings": step_learnings,
                "urls": step_urls
            })

        # Perform final analysis
        steps_summary_text = "\n\n".join([
            f"Step {s['step_id']}: {s['description']}\nLearnings: {json.dumps(s['learnings'])}"
            for s in step_summaries
        ])

        final_prompt = RESEARCH_SUMMARY_PROMPT.format(
            query=refined_query,
            steps_summary=steps_summary_text
        )

        final_result = await generate_json_completion(final_prompt, temperature=0.7)
        logger.debug(f"Final research result: {final_result}")

        # Add final findings to learnings
        if "findings" in final_result:
            all_learnings.extend(final_result["findings"])

        # Return compiled results
        return {
            "query": refined_query,
            "originalQuery": query,
            "learnings": list(set(all_learnings)),  # Remove duplicates
            "visitedUrls": list(set(visited_urls)),  # Remove duplicates
            "summary": final_result
        }

    except Exception as e:
        logger.error(f"Error in deep research: {str(e)}")
        logger.error(traceback.format_exc())

        return {
            "error": str(e),
            "learnings": all_learnings,
            "visitedUrls": visited_urls
        }


def deep_research_sync(
        query: str,
        breadth: int = None,
        depth: int = None,
        on_progress: Optional[Callable] = None,
        user_clarifications: Dict[str, str] = None,
        search_source: Optional[str] = None,
        history_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the deep_research function.
    
    Args:
        query: The research query
        breadth: Number of parallel queries to run at each step
        depth: Number of iterations to perform
        on_progress: Optional callback function for progress updates
        user_clarifications: User responses to clarification questions
        search_source: Optional search provider to use
        
    Returns:
        Dict with research results
    """
    # Add event loop policy for Windows if needed
    add_event_loop_policy()

    # Run the async function in the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        deep_research(
            query=query,
            breadth=breadth,
            depth=depth,
            on_progress=on_progress,
            user_clarifications=user_clarifications,
            search_source=search_source,
            history_context=history_context
        )
    )


async def write_final_report(query: str, learnings: List[str], visited_urls: List[str], history_context: str = '') -> str:
    """
    Generate a final research report based on learnings and sources.
    
    Args:
        query: The original research query
        learnings: List of key learnings/facts discovered
        visited_urls: List of sources consulted
        history_context: chat history
        
    Returns:
        A formatted research report
    """
    formatted_prompt = FINAL_REPORT_PROMPT.format(
        prompt=query,
        learnings=' '.join(learnings),
        sources=' '.join(visited_urls),
        history_context=history_context
    )

    report = await generate_completion(
        prompt=formatted_prompt,
        system_message=SYSTEM_PROMPT,
        temperature=0.7
    )

    return report


async def write_final_answer(query: str, learnings: List[str], history_context: str = '') -> str:
    """
    Generate a final concise answer based on the research.
    
    Args:
        query: The original research query
        learnings: List of key learnings/facts discovered
        
    Returns:
        A concise answer to the query
    """
    formatted_prompt = FINAL_ANSWER_PROMPT.format(
        prompt=query,
        learnings=' '.join(learnings),
        history_context=history_context
    )

    answer = await generate_completion(
        prompt=formatted_prompt,
        system_message=SYSTEM_PROMPT,
        temperature=0.7
    )

    return answer


def write_final_report_sync(query: str, learnings: List[str], visited_urls: List[str], history_context: str = '') -> str:
    """
    Synchronous wrapper for the write_final_report function.
    
    Args:
        query: The original research query
        learnings: List of key learnings/facts discovered
        visited_urls: List of sources consulted
        
    Returns:
        A formatted research report
    """
    # Add event loop policy for Windows if needed
    add_event_loop_policy()

    # Run the async function in the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        write_final_report(
            query=query,
            learnings=learnings,
            visited_urls=visited_urls,
            history_context=history_context
        )
    )


def write_final_answer_sync(query: str, learnings: List[str], history_context: str = '') -> str:
    """
    Synchronous wrapper for the write_final_answer function.
    
    Args:
        query: The original research query
        learnings: List of key learnings/facts discovered
        
    Returns:
        A concise answer to the query
    """
    # Add event loop policy for Windows if needed
    add_event_loop_policy()

    # Run the async function in the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        write_final_answer(
            query=query,
            learnings=learnings,
            history_context=history_context
        )
    )


async def deep_research_stream(
        query: str,
        breadth: int = None,
        depth: int = None,
        on_progress: Optional[Callable] = None,
        user_clarifications: Dict[str, str] = None,
        search_source: Optional[str] = None,
        history_context: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming version of deep research that yields partial results.
    
    Args:
        query: The research query
        breadth: Number of parallel queries to run at each step
        depth: Number of iterations to perform
        on_progress: Optional callback function for progress updates
        user_clarifications: User responses to clarification questions
        search_source: Optional search provider to use
        history_context: history chat content
        
    Yields:
        Dict with partial research results and status updates
    """
    # Load configuration
    config = get_config()

    # Use defaults from config if not specified
    if breadth is None:
        breadth = config["research"]["default_breadth"]
    if depth is None:
        depth = config["research"]["default_depth"]

    # Initialize tracking variables
    visited_urls = []
    all_learnings = []

    # Initialize search provider
    search_provider = get_search_provider(search_source=search_source, context=SearchContext())

    try:
        # Step 1: Yield initial status
        yield {
            "status_update": f"开始研究查询: '{query}'...",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "current_query": query,
            "stage": "initial"
        }

        # Step 1.5: 先判断是否需要生成澄清问题
        needs_clarification = False

        # 如果没有提供用户澄清，则判断是否需要澄清
        if not user_clarifications:
            yield {
                "status_update": f"分析查询是否需要澄清...",
                "learnings": all_learnings,
                "visitedUrls": visited_urls,
                "current_query": query,
                "stage": "analyzing_query"
            }
            
            # 修复重复调用问题 - 使用缓存变量存储结果
            _clarification_cache_key = f"should_clarify_{query}"
            if _clarification_cache_key not in globals():
                needs_clarification = await should_clarify_query(query, history_context)
                globals()[_clarification_cache_key] = needs_clarification
            else:
                needs_clarification = globals()[_clarification_cache_key]
                logger.debug(f"使用缓存的澄清结果: {query}, result: {needs_clarification}")

            if not needs_clarification:
                yield {
                    "status_update": f"查询已足够清晰，跳过澄清步骤",
                    "learnings": all_learnings,
                    "visitedUrls": visited_urls,
                    "current_query": query,
                    "stage": "clarification_skipped"
                }

        # 如果LLM判断需要澄清，或者已经有用户澄清，则继续生成或处理澄清问题
        questions = []
        if needs_clarification or user_clarifications:
            # Step 2: Generate clarification questions if needed
            if needs_clarification:
                yield {
                    "status_update": f"生成澄清问题...",
                    "learnings": all_learnings,
                    "visitedUrls": visited_urls,
                    "current_query": query,
                    "stage": "generating_questions"
                }

                followup_result = await generate_followup_questions(query, history_context)
                questions = followup_result.get("questions", [])

                if questions:
                    # If clarification is needed, update status
                    yield {
                        "status_update": f"查询需要澄清，生成了 {len(questions)} 个问题",
                        "learnings": all_learnings,
                        "visitedUrls": visited_urls,
                        "current_query": query,
                        "questions": questions,
                        "stage": "clarification_needed"
                    }

                    # If we don't have user responses yet, wait for them
                    if not user_clarifications:
                        yield {
                            "status_update": "等待用户回答澄清问题...",
                            "learnings": all_learnings,
                            "visitedUrls": visited_urls,
                            "current_query": query,
                            "questions": questions,
                            "awaiting_clarification": True,
                            "stage": "awaiting_clarification"
                        }
                        return

        # Step 3: Process user clarifications if provided
        refined_query = query
        user_responses = user_clarifications or {}

        if questions and user_clarifications:
            # Track which questions were answered vs. which use defaults
            answered_questions = []
            unanswered_questions = []

            for q in questions:
                key = q.get("key", "")
                question_text = q.get("question", "")
                if key in user_responses and user_responses[key]:
                    answered_questions.append(question_text)
                else:
                    unanswered_questions.append(question_text)

            yield {
                "status_update": f"处理用户的澄清回答 ({len(answered_questions)}/{len(questions)} 已回答)",
                "learnings": all_learnings,
                "visitedUrls": visited_urls,
                "current_query": query,
                "answered_questions": answered_questions,
                "unanswered_questions": unanswered_questions,
                "stage": "processing_clarifications"
            }

            # Process the clarifications
            clarification_result = await process_clarifications(query, user_responses, questions, history_context)
            refined_query = clarification_result.get("refined_query", query)

            yield {
                "status_update": f"查询已优化: '{refined_query}'",
                "learnings": all_learnings,
                "visitedUrls": visited_urls,
                "original_query": query,
                "current_query": refined_query,
                "assumptions": clarification_result.get("assumptions", []),
                "stage": "query_refined"
            }

            # Check if this is a simple query that can be answered directly
            if not clarification_result.get("requires_search", True):
                direct_answer = clarification_result.get("direct_answer", "")
                if direct_answer:
                    yield {
                        "status_update": "查询可以直接回答，无需搜索",
                        "requires_search": False,
                        "direct_answer": direct_answer,
                        "final_report": direct_answer,  # 直接使用direct_answer作为最终报告
                        "learnings": ["直接回答: " + direct_answer],
                        "visitedUrls": [],
                        "stage": "completed"
                    }
                    return  # 不再需要继续执行查询流程

        # Step 4: Generate research plan with variable steps
        yield {
            "status_update": f"为查询生成研究计划: '{refined_query}'",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "current_query": refined_query,
            "stage": "planning"
        }

        plan_result = await generate_research_plan(refined_query, history_context)
        steps = plan_result.get("steps", [])

        yield {
            "status_update": f"生成了 {len(steps)} 步研究计划: {plan_result.get('assessments', '无评估')}",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "current_query": refined_query,
            "research_plan": steps,
            "stage": "plan_generated"
        }

        # Track step summaries for the final analysis
        step_summaries = []

        # Iterate through each step in the research plan
        for step_idx, step in enumerate(steps):
            step_id = step.get("step_id", step_idx + 1)
            description = step.get("description", f"Research step {step_id}")
            search_query = step.get("search_query", refined_query)

            yield {
                "status_update": f"开始研究步骤 {step_id}/{len(steps)}: {description}",
                "learnings": all_learnings,
                "visitedUrls": visited_urls,
                "current_query": refined_query,
                "step_query": search_query,
                "current_step": step,
                "progress": {
                    "current_step": step_id,
                    "total_steps": len(steps)
                },
                "stage": "step_started"
            }

            # Create a queue of queries to process for this step
            queries_to_process = [search_query]
            step_urls = []
            step_learnings = []

            # Process the initial query and follow-ups
            current_depth = depth
            total_processed = 0

            while queries_to_process and current_depth > 0:
                # Get the next queries to process (up to breadth)
                current_queries = queries_to_process[:breadth]
                remaining_queries = len(queries_to_process) - len(current_queries)
                queries_to_process = queries_to_process[breadth:]

                # Research these queries concurrently
                yield {
                    "status_update": f"步骤 {step_id}/{len(steps)}: 并行研究 {len(current_queries)} 个查询，剩余队列: {remaining_queries}",
                    "learnings": all_learnings,
                    "visitedUrls": visited_urls,
                    "current_queries": current_queries,
                    "depth_level": current_depth,
                    "max_depth": depth,
                    "progress": {
                        "current_step": step_id,
                        "total_steps": len(steps),
                        "current_depth": current_depth,
                        "max_depth": depth,
                        "processed_queries": total_processed,
                        "remaining_queries": remaining_queries
                    },
                    "stage": "processing_queries"
                }

                # Process each query in the current batch
                research_tasks = []
                for current_query in current_queries:
                    task = research_step(
                        query=current_query,
                        depth=current_depth,
                        breadth=breadth,
                        config=config,
                        on_progress=on_progress,
                        search_provider=search_provider,
                        step_info=step,
                        all_learnings=all_learnings
                    )
                    research_tasks.append(task)

                # Execute tasks with concurrency
                results = await asyncio.gather(*research_tasks)

                # Process the results
                new_learnings = []
                new_urls = []
                for result in results:
                    # Update tracking variables
                    step_urls.extend(result["urls"])
                    new_urls.extend(result["urls"])
                    step_learnings.extend(result["learnings"])
                    new_learnings.extend(result["learnings"])
                    total_processed += 1

                    # Add the new queries to our queue
                    next_queries = result["nextQueries"]
                    queries_to_process.extend(next_queries)

                # Decrement depth for next iteration
                current_depth -= 1

                # Format learnings and URLs for display
                formatted_learnings = []
                for i, learning in enumerate(new_learnings):
                    formatted_learnings.append(f"[{i + 1}] {learning}")

                formatted_urls = []
                for i, url in enumerate(new_urls):
                    formatted_urls.append(f"[{i + 1}] {url}")

                # Yield intermediate results
                yield {
                    "status_update": f"步骤 {step_id}/{len(steps)}: 发现 {len(new_learnings)} 个新见解",
                    "learnings": all_learnings + step_learnings,
                    "visitedUrls": visited_urls + step_urls,
                    "new_learnings": new_learnings,
                    "formatted_new_learnings": formatted_learnings,
                    "new_urls": new_urls,
                    "formatted_new_urls": formatted_urls,
                    "next_queries": queries_to_process[:3],  # Show a few upcoming queries
                    "progress": {
                        "current_step": step_id,
                        "total_steps": len(steps),
                        "current_depth": current_depth,
                        "max_depth": depth,
                        "processed_queries": total_processed
                    },
                    "stage": "insights_found"
                }

            # Update visited URLs and learnings
            visited_urls.extend(step_urls)
            all_learnings.extend(step_learnings)

            # Save step summary
            step_summaries.append({
                "step_id": step_id,
                "description": description,
                "learnings": step_learnings,
                "urls": step_urls
            })

            # Format all step learnings for display
            formatted_step_learnings = []
            for i, learning in enumerate(step_learnings):
                formatted_step_learnings.append(f"[{i + 1}] {learning}")

            # Step completion update
            yield {
                "status_update": f"完成研究步骤 {step_id}/{len(steps)}: {description}，获得 {len(step_learnings)} 个见解",
                "learnings": all_learnings,
                "visitedUrls": visited_urls,
                "step_learnings": step_learnings,
                "formatted_step_learnings": formatted_step_learnings,
                "step_urls": step_urls,
                "progress": {
                    "current_step": step_id,
                    "total_steps": len(steps),
                    "completed": True
                },
                "stage": "step_completed"
            }

        # Perform final analysis
        yield {
            "status_update": "分析所有已收集的信息...",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "stage": "final_analysis"
        }

        steps_summary_text = "\n\n".join([
            f"步骤 {s['step_id']}: {s['description']}\n发现: {json.dumps(s['learnings'], ensure_ascii=False)}"
            for s in step_summaries
        ])

        final_prompt = RESEARCH_SUMMARY_PROMPT.format(
            query=refined_query,
            steps_summary=steps_summary_text
        )

        final_result = await generate_json_completion(final_prompt, temperature=0.7)

        # Add final findings to learnings
        final_findings = []
        if "findings" in final_result:
            all_learnings.extend(final_result["findings"])
            final_findings = final_result["findings"]

        # Format final findings
        formatted_findings = []
        for i, finding in enumerate(final_findings):
            formatted_findings.append(f"[{i + 1}] {finding}")

        yield {
            "status_update": f"分析完成，得出 {len(final_findings)} 个主要发现",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "final_findings": final_findings,
            "formatted_final_findings": formatted_findings,
            "gaps": final_result.get("gaps", []),
            "recommendations": final_result.get("recommendations", []),
            "stage": "analysis_completed"
        }
        final_answer = ""
        yield {
            "status_update": "生成详细研究报告...",
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "final_answer": final_answer,
            "stage": "generating_report"
        }

        # Generate report (non-streaming for now)
        final_report = await write_final_report(refined_query, all_learnings, visited_urls, history_context)

        # Return compiled results
        yield {
            "status_update": "研究完成!",
            "query": refined_query,
            "originalQuery": query,
            "learnings": list(set(all_learnings)),  # Remove duplicates
            "visitedUrls": list(set(visited_urls)),  # Remove duplicates
            "summary": final_result,
            "final_answer": final_answer,
            "final_report": final_report,
            "stage": "completed"
        }

    except Exception as e:
        logger.error(f"Error in deep research stream: {str(e)}")
        logger.error(traceback.format_exc())

        yield {
            "status_update": f"错误: {str(e)}",
            "error": str(e),
            "learnings": all_learnings,
            "visitedUrls": visited_urls,
            "stage": "error"
        }


async def write_final_report_stream(prompt: str, learnings: List[str], visited_urls: List[str], history_context: str = '') -> AsyncGenerator[
    str, None]:
    """
    Streaming version of write_final_report that yields chunks of the report.
    
    Args:
        prompt: The original research query
        learnings: List of key learnings/facts discovered
        visited_urls: List of sources consulted
        
    Yields:
        Chunks of the final report
    """
    formatted_prompt = FINAL_REPORT_PROMPT.format(
        prompt=prompt,
        learnings=' '.join(learnings),
        sources=' '.join(visited_urls),
        history_context=history_context
    )

    response_generator = await generate_completion(
        prompt=formatted_prompt,
        system_message="You are an expert researcher providing detailed, well-structured reports in Chinese.",
        temperature=0.7,
        stream=True
    )

    # Stream the response chunks
    async for chunk in response_generator:
        yield chunk


async def write_final_answer_stream(prompt: str, learnings: List[str], history_context: str = '') -> AsyncGenerator[str, None]:
    """
    Streaming version of write_final_answer that yields chunks of the answer.
    
    Args:
        prompt: The original research query
        learnings: List of key learnings/facts discovered
        
    Yields:
        Chunks of the final answer
    """
    formatted_prompt = FINAL_ANSWER_PROMPT.format(
        prompt=prompt,
        learnings=' '.join(learnings),
        history_context=history_context
    )

    response_generator = await generate_completion(
        prompt=formatted_prompt,
        system_message="You are an expert researcher providing concise answers based on research findings in Chinese.",
        temperature=0.7,
        stream=True
    )

    # Stream the response chunks
    async for chunk in response_generator:
        yield chunk
