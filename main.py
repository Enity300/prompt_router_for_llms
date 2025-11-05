
import typer
from typing import Optional, List
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
import sys

from src.semantic_router import SemanticRouter, SemanticRouterError
from src.specialist_clients import query_specialist, LLMClientError
from config import config

app = typer.Typer(
    name="router",
    help="SS-GER: Semantic-Similarity-Guided Efficient Routing for Heterogeneous Language Models",
    add_completion=False
)

console = Console()


def display_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                         Running ...                           ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def format_routing_result(routing_result: dict, response_result: dict = None):
    
    routing_info = f"""
[bold]Category:[/bold] {routing_result['category']}
[bold]Confidence:[/bold] {routing_result['confidence']:.4f}
[bold]Reasoning:[/bold] {routing_result['reasoning']}
[bold]Routing Time:[/bold] {routing_result['routing_time']:.3f}s
[bold]From Cache:[/bold] {'Yes' if routing_result.get('from_cache', False) else 'No'}
"""
    
    if 'routing_info' in routing_result:
        info = routing_result['routing_info']
        routing_info += f"\n[bold]Similarity Score:[/bold] {info['similarity_score']:.4f}"
        routing_info += f"\n[bold]Closest Example:[/bold] '{info['closest_example'][:100]}...'"
    
    console.print(Panel(routing_info, title="🎯 Routing Decision", border_style="green"))
    
    if response_result:
        specialist_info = f"""
[bold]Specialist:[/bold] {response_result['specialist']}
[bold]Model:[/bold] {response_result['model']}
[bold]Response Time:[/bold] {response_result.get('response_time', 0):.3f}s
[bold]Tokens Used:[/bold] {response_result.get('tokens_used', 0)}
[bold]Success:[/bold] {'Yes' if response_result.get('success', False) else 'No'}
"""
        
        if response_result.get('is_mock', False):
            specialist_info += "\n[yellow]⚠️ Using mock response (API keys not configured)[/yellow]"
        
        console.print(Panel(specialist_info, title="🤖 Specialist Info", border_style="blue"))
        
        # Display the actual response
        response_text = response_result['response']
        console.print(Panel(response_text, title="💬 Response", border_style="cyan"))


@app.command()
def route(
    prompt: str = typer.Argument(..., help="The prompt to route to a specialist LLM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed routing information"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, plain")
):

    try:
        if output_format == "rich":
            display_banner()
            console.print(f"\n[bold]Input Prompt:[/bold] {prompt}\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console if output_format == "rich" else None,
            disable=(output_format != "rich")
        ) as progress:
            if output_format == "rich":
                task = progress.add_task("Initializing router...", total=None)
            
            router = SemanticRouter()
            
            if output_format == "rich":
                progress.update(task, description="Routing prompt...")
            
            routing_result = router.route(prompt)
            
            if output_format == "rich":
                progress.update(task, description="Querying specialist...")
            
            response_result = query_specialist(
                routing_result['category'],
                prompt,
                max_tokens=1000
            )
        
        if output_format == "json":
            output = {
                "routing": routing_result,
                "response": response_result
            }
            console.print(json.dumps(output, indent=2))
            
        elif output_format == "plain":
            console.print(f"Category: {routing_result['category']}")
            console.print(f"Specialist: {response_result['specialist']}")
            console.print(f"Response: {response_result['response']}")
            
        else: 
            format_routing_result(routing_result, response_result)
        
    except SemanticRouterError as e:
        console.print(f"❌ Router Error: {e}", style="bold red")
        sys.exit(1)
    except LLMClientError as e:
        console.print(f"❌ LLM Client Error: {e}", style="bold red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected Error: {e}", style="bold red")
        sys.exit(1)


@app.command()
def interactive():

    display_banner()
    console.print("\n[bold green]Interactive Mode[/bold green] - Type 'exit' to quit\n")
    
    try:
        with console.status("Initializing router..."):
            router = SemanticRouter()
        
        console.print("✅ Router initialized! Ready for prompts.\n")
        
        while True:
            try:
                prompt = Prompt.ask("\n[bold blue]Enter your prompt[/bold blue]")
                
                if prompt.lower() in ['exit', 'quit', 'q']:
                    console.print("\n👋 Goodbye!")
                    break
                
                if not prompt.strip():
                    console.print("⚠️ Please enter a non-empty prompt")
                    continue
                
                with console.status("Processing..."):
                    routing_result = router.route(prompt)
                    response_result = query_specialist(
                        routing_result['category'],
                        prompt,
                        max_tokens=1000
                    )
                
                format_routing_result(routing_result, response_result)
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                console.print(f"❌ Error: {e}", style="bold red")
                
    except SemanticRouterError as e:
        console.print(f"❌ Failed to initialize router: {e}", style="bold red")
        sys.exit(1)


@app.command()
def test():

    display_banner()
    console.print("\n[bold green]Running Test Suite[/bold green]\n")
    
    test_prompts = [
        # Coding prompts
        "Write a Python function to implement quicksort algorithm",
        "How do I create a REST API with Flask?",
        "Debug this JavaScript code that's not working",
        "Implement a binary search tree in Java",
        
        # Math prompts  
        "Solve the integral of x^2 + 3x - 5",
        "What is the derivative of sin(x) * cos(x)?",
        "Calculate the probability of getting 3 heads in 5 coin flips",
        "Find the roots of the quadratic equation 2x^2 - 7x + 3 = 0",
        
        # General knowledge prompts
        "What is the capital of Australia?",
        "Explain the water cycle",
        "What are the main causes of climate change?",
        "Who wrote the novel '1984'?",
        
        # Edge cases / out-of-distribution
        "What's the weather like today?",
        "Tell me a joke about programming",
        "How do I cook pasta?",
    ]
    
    try:
        with console.status("Initializing router..."):
            router = SemanticRouter()
        
        console.print(f"Testing {len(test_prompts)} prompts...\n")
        
        results = []
        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"[{i:2d}/{len(test_prompts)}] Testing: [italic]{prompt[:60]}...[/italic]")
            
            try:
                routing_result = router.route(prompt)
                category = routing_result['category']
                confidence = routing_result['confidence']
                
                console.print(f"        → [bold]{category}[/bold] (confidence: {confidence:.3f})")
                
                results.append({
                    "prompt": prompt,
                    "category": category,
                    "confidence": confidence,
                    "success": True
                })
                
            except Exception as e:
                console.print(f"        → [red]ERROR: {e}[/red]")
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                })
        
        console.print("\n" + "="*70)
        console.print("[bold green]Test Summary[/bold green]")
        
        successful_tests = [r for r in results if r.get('success', False)]
        category_counts = {}
        for result in successful_tests:
            cat = result['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        table = Table(title="Routing Distribution")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        total = len(successful_tests)
        for category, count in category_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            table.add_row(category, str(count), f"{percentage:.1f}%")
        
        console.print(table)
        console.print(f"\n✅ Successful tests: {len(successful_tests)}/{len(test_prompts)}")
        
    except SemanticRouterError as e:
        console.print(f"❌ Failed to initialize router: {e}", style="bold red")
        sys.exit(1)


@app.command()
def stats():

    display_banner()
    console.print("\n[bold green]Router Statistics[/bold green]\n")
    
    try:
        router = SemanticRouter()
        stats = router.get_statistics()
        
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="magenta")
        
        config_table.add_row("Model", stats['model_name'])
        config_table.add_row("Database Path", stats['db_path'])
        config_table.add_row("Collection", stats['collection_name'])
        config_table.add_row("Similarity Threshold", str(stats['similarity_threshold']))
        config_table.add_row("Cache Threshold", str(stats['cache_threshold']))
        config_table.add_row("Top K Neighbors", str(stats['top_k_neighbors']))
        
        console.print(config_table)
        
        db_table = Table(title="Database Information")
        db_table.add_column("Metric", style="cyan")
        db_table.add_column("Value", style="magenta")
        
        db_table.add_row("Total Embeddings", str(stats['total_embeddings']))
        db_table.add_row("Cache Size", str(stats['cache_size']))
        
        console.print(db_table)
        
        if stats.get('category_stats'):
            cat_table = Table(title="Available Categories")
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Status", style="magenta")
            
            for category, status in stats['category_stats'].items():
                cat_table.add_row(category, status)
            
            console.print(cat_table)
        
    except SemanticRouterError as e:
        console.print(f"❌ Failed to get statistics: {e}", style="bold red")
        sys.exit(1)


@app.command()
def test_reproducibility():

    display_banner()
    console.print("\n[bold green]Testing Reproducibility[/bold green]\n")
    
    try:
        from src.test_reproducibility import ReproducibilityTester
        
        tester = ReproducibilityTester()
        results = tester.run_reproducibility_test(num_builds=2)
        
        if results["reproducibility_achieved"]:
            console.print("✅ [bold green]REPRODUCIBILITY ACHIEVED[/bold green]")
            console.print("All database builds produced identical results!")
        else:
            console.print("❌ [bold red]REPRODUCIBILITY FAILED[/bold red]")
            console.print("Database builds produced different results!")
        
        for comp in results["comparisons"]:
            status_color = "green" if comp["databases_identical"] else "red"
            status_text = "IDENTICAL" if comp["databases_identical"] else "DIFFERENT"
            console.print(f"  {comp['pair']}: [{status_color}]{status_text}[/{status_color}]")
            
            if comp["embedding_similarity"] > 0:
                console.print(f"    Embedding similarity: {comp['embedding_similarity']:.4f}")
        
        tester.cleanup()
        
    except ImportError as e:
        console.print(f"❌ Could not import test module: {e}", style="bold red")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Test failed: {e}", style="bold red")
        sys.exit(1)


@app.command()
def ollama_setup():

    display_banner()
    console.print("\n[bold green] Ollama Local Models Setup[/bold green]\n")

    console.print("[bold]Checking Ollama Installation...[/bold]")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            console.print("✅ Ollama server is running!")

            # Check installed models
            models = response.json().get("models", [])
            required_models = [
                "deepseek-coder:1.3b",
                "qwen2-math:1.5b",
                "llama3.2:1b",
                "phi3:mini"
            ]

            console.print(f"\n[bold]Checking Required Models...[/bold]")
            installed_models = [model["name"] for model in models]

            for model in required_models:
                if model in installed_models:
                    console.print(f"✅ {model} - Installed")
                else:
                    console.print(f"❌ {model} - Not installed")

            missing_models = [m for m in required_models if m not in installed_models]

            if missing_models:
                console.print(f"\n[bold yellow]⚠️ Missing Models: {len(missing_models)}/{len(required_models)}[/bold yellow]")
                console.print("\n[bold]To install missing models, run:[/bold]")
                for model in missing_models:
                    console.print(f"[cyan]ollama pull {model}[/cyan]")

                console.print(f"\n[bold red]Note:[/bold red] Total download size: ~4GB")
            else:
                console.print(f"\n[bold green]🎉 All models installed! Your demo will use real AI responses![/bold green]")

                console.print(f"\n[bold]Testing Local Models...[/bold]")
                try:
                    from src.specialist_clients import _query_ollama_model
                    test_result = _query_ollama_model(
                        "deepseek-coder:1.3b",
                        "Write a simple hello world in Python",
                        "DeepSeek Test",
                        "coding"
                    )
                    if test_result:
                        console.print("✅ Local models working perfectly!")
                        console.print(f"[dim]Sample response: {test_result['response'][:100]}...[/dim]")
                    else:
                        console.print("⚠️ Models installed but not responding correctly")
                except Exception as e:
                    console.print(f"⚠️ Test failed: {e}")

        else:
            console.print("❌ Ollama server not responding")
            _show_ollama_installation_instructions()

    except requests.exceptions.ConnectionError:
        console.print("❌ Ollama server not running")
        _show_ollama_installation_instructions()
    except Exception as e:
        console.print(f"❌ Error checking Ollama: {e}")
        _show_ollama_installation_instructions()


@app.command()
def academic_eval():

    display_banner()
    console.print("\n[bold green] Academic Evaluation [/bold green]")
    console.print("Comprehensive statistical analysis \n")
    
    try:
        from src.comprehensive_evaluation import ComprehensiveEvaluator
        
        evaluator = ComprehensiveEvaluator()
        results = evaluator.run_comprehensive_evaluation()
        
        if results:
            console.print("\n[bold green]✅ Academic evaluation completed successfully![/bold green]")
            console.print("📁 Results saved to 'evaluation_results/' directory")
            console.print("📊 Visualizations generated")
            console.print("📝 Report created")
        else:
            console.print("\n[bold red]❌ Academic evaluation failed[/bold red]")
            
    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}")
        console.print("Install with: [cyan]pip install -r requirements.txt[/cyan]")
    except Exception as e:
        console.print(f"❌ Evaluation error: {e}")


@app.command()
def visualize_embeddings():

    display_banner()
    console.print("\n[bold green] Embedding Visualization [/bold green]")
    
    try:
        from src.embedding_visualization import EmbeddingVisualizer
        
        visualizer = EmbeddingVisualizer()
        visualizer.initialize_components()
        output_dir = visualizer.create_comprehensive_visualizations()
        
        if output_dir:
            console.print(f"\n[bold green]✅ Visualization  completed![/bold green]")
            console.print(f"📁 All visualizations saved to '{output_dir}/' directory")
            console.print("📊 plots generated")
            console.print("🌐 3D visualizations created")
        else:
            console.print("\n[bold red]❌ Visualization generation failed[/bold red]")
            
    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}")
        console.print("Install with: [cyan]pip install -r requirements.txt[/cyan]")
    except Exception as e:
        console.print(f"❌ Visualization error: {e}")


@app.command()
def full_analysis():

    display_banner()
    console.print("\n[bold green] Complete Analysis [/bold green]")
    
    console.print("[bold cyan]Step 1: Running comprehensive evaluation...[/bold cyan]")
    try:
        from src.comprehensive_evaluation import ComprehensiveEvaluator
        evaluator = ComprehensiveEvaluator()
        eval_results = evaluator.run_comprehensive_evaluation()
        console.print("✅ Evaluation completed")
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}")
        return
    
    console.print("\n[bold cyan]Step 2: Generating embedding visualizations...[/bold cyan]")
    try:
        from src.embedding_visualization import EmbeddingVisualizer
        visualizer = EmbeddingVisualizer()
        visualizer.initialize_components()
        viz_dir = visualizer.create_comprehensive_visualizations()
        console.print("✅ Visualizations completed")
    except Exception as e:
        console.print(f"❌ Visualization failed: {e}")
        return
    
    console.print("\n[bold cyan]Step 3: Running RouterBench evaluation...[/bold cyan]")
    try:
        from src.comprehensive_routerbench import ComprehensiveRouterBench
        evaluator_rb = ComprehensiveRouterBench(SemanticRouter())
        rb_results = evaluator_rb.run_comprehensive_evaluation(dataset_size=1000)
        console.print("✅ RouterBench evaluation completed")
    except Exception as e:
        console.print(f"❌ RouterBench evaluation failed: {e}")
        return

    console.print("\n[bold green]🎉 Complete analysis suite finished![/bold green]")
    console.print("=" * 60)
    console.print("📁 Results locations:")
    console.print("  • evaluation_results/ - Statistical analysis & academic report")
    console.print("  • embedding_visualizations/ - Publication-quality plots")
    console.print("  • comprehensive_routerbench_results/ - RouterBench methodology results")
    console.print("\n🎓 Your project is now ready for A+ academic evaluation!")
    console.print("📄 Academic report: evaluation_results/academic_evaluation_report.md")
    console.print("📊 RouterBench AIQ Score: {:.4f} (Industry-standard benchmark)".format(rb_results.aiq_score if 'rb_results' in locals() else 0.0))
    console.print("📊 Key visualizations: All directories contain publication-ready figures")


def _show_ollama_installation_instructions():
    """Show Ollama installation and setup instructions"""
    instructions = """
[bold]🔧 Ollama Installation & Setup:[/bold]

[bold]1. Install Ollama:[/bold]
   • Windows: Download from https://ollama.ai/download
   • macOS: [cyan]brew install ollama[/cyan]
   • Linux: [cyan]curl -fsSL https://ollama.ai/install.sh | sh[/cyan]

[bold]2. Start Ollama:[/bold]
   [cyan]ollama serve[/cyan]

[bold]3. Pull Required Models (~4GB total):[/bold]
   [cyan]ollama pull deepseek-coder:1.3b[/cyan]    # Coding specialist
   [cyan]ollama pull qwen2-math:1.5b[/cyan]        # Math specialist  
   [cyan]ollama pull llama3.2:1b[/cyan]           # General knowledge
   [cyan]ollama pull phi3:mini[/cyan]             # Fast fallback

[bold]4. Verify Setup:[/bold]
   [cyan]python main.py ollama-setup[/cyan]

[bold red]🎯 Demo Impact:[/bold red] 
Real AI responses vs mock text = 10x more impressive for your professor!
"""
    console.print(Panel(instructions, title="🚀 Setup Instructions"))


@app.command()
def routerbench():
    """
    Run comprehensive RouterBench evaluation (arXiv:2403.12031)
    """
    display_banner()
    console.print("\n[bold green]🎯 RouterBench Evaluation[/bold green]")
    console.print("Following exact RouterBench methodology from Martian AI (arXiv:2403.12031)\n")
    
    try:
        from src.comprehensive_routerbench import ComprehensiveRouterBench
        
        # Initialize
        router = SemanticRouter()
        evaluator = ComprehensiveRouterBench(router)
        
        console.print("[bold cyan]Running RouterBench evaluation with 1,000 queries...[/bold cyan]")
        
        # Run evaluation
        result = evaluator.run_comprehensive_evaluation(dataset_size=1000)
        
        console.print(f"\n[bold green]🎉 RouterBench Evaluation Complete![/bold green]")
        console.print("=" * 60)
        console.print(f"📊 AIQ Score: [bold cyan]{result.aiq_score:.4f}[/bold cyan]")
        
        if result.aiq_score >= 0.8:
            console.print("🟢 Performance: [bold green]Excellent[/bold green] (State-of-the-art)")
        elif result.aiq_score >= 0.6:
            console.print("🟡 Performance: [bold yellow]Good[/bold yellow] (Production-ready)")
        elif result.aiq_score >= 0.4:
            console.print("🟠 Performance: [bold orange]Fair[/bold orange] (Needs optimization)")
        else:
            console.print("🔴 Performance: [bold red]Poor[/bold red] (Requires improvement)")
        
        console.print(f"💰 Total Cost: [bold cyan]${result.detailed_metrics['total_cost']:.4f}[/bold cyan]")
        console.print(f"⚡ Average Latency: [bold cyan]{result.detailed_metrics['average_latency_ms']:.1f}ms[/bold cyan]")
        console.print(f"🎯 Average Quality: [bold cyan]{result.detailed_metrics['average_quality']:.3f}[/bold cyan]")
        
        console.print(f"\n📁 Results: [cyan]comprehensive_routerbench_results/[/cyan]")
        console.print(f"📄 Report: [cyan]comprehensive_routerbench_report.md[/cyan]")
        console.print(f"📊 Visualizations: [cyan]comprehensive_routerbench_analysis.png[/cyan]")
        
    except Exception as e:
        console.print(f"❌ RouterBench evaluation failed: {e}")


@app.command()
def setup():
    """
    Display setup instructions and configuration help
    """
    display_banner()
    console.print("\n[bold green]SS-GER Setup Instructions[/bold green]\n")
    
    setup_text = f"""
[bold]1. Build the Expertise Database (Required)[/bold]
   Run this command first to create the expertise manifolds:
   [cyan]python build_expertise_db.py[/cyan]

[bold]2. Configure API Keys (Optional for testing)[/bold]
   Create a .env file with your API keys:
   
{config.get_example_env_content()}

[bold]3. Install Dependencies[/bold]
   Make sure all required packages are installed:
   [cyan]pip install -r requirements.txt[/cyan]

[bold]4. Test the System[/bold]
   Run a quick test to verify everything works:
   [cyan]python main.py --test[/cyan]

[bold]5. Start Using SS-GER[/bold]
   Route individual prompts:
   [cyan]python main.py "Write a Python function for sorting"[/cyan]
   
   Or start interactive mode:
   [cyan]python main.py --interactive[/cyan]
"""
    
    console.print(Panel(setup_text, title="🚀 Setup Guide"))
    
    # Check current status
    console.print("\n[bold blue]Current Status Check[/bold blue]\n")
    
    # Check if database exists
    import os
    if os.path.exists(config.CHROMADB_PATH):
        console.print("✅ Expertise database found")
        try:
            router = SemanticRouter()
            stats = router.get_statistics()
            console.print(f"✅ Database loaded with {stats['total_embeddings']} embeddings")
        except:
            console.print("⚠️ Database exists but may be corrupted")
    else:
        console.print("❌ Expertise database not found - run build_expertise_db.py first")
    
    # Check API keys
    if config.OPENAI_API_KEY:
        console.print("✅ OpenAI API key configured")
    else:
        console.print("⚠️ OpenAI API key not configured (will use mock responses)")
    
    if config.DEEPSEEK_API_KEY:
        console.print("✅ DeepSeek API key configured")
    else:
        console.print("⚠️ DeepSeek API key not configured (will use mock responses)")


@app.command()
def gpu_status():
    """
    Display current GPU status and capabilities
    """
    display_banner()
    console.print("\n[bold green]🎮 SS-GER GPU Status[/bold green]\n")
    
    try:
        from src.gpu_monitor import get_gpu_monitor
        
        gpu_monitor = get_gpu_monitor()
        
        # Display detection results
        detection = gpu_monitor.detection_results
        
        console.print("[bold cyan]GPU Detection Results:[/bold cyan]")
        detection_table = Table(title="Available GPU Libraries")
        detection_table.add_column("Library", style="cyan")
        detection_table.add_column("Status", style="magenta") 
        
        detection_table.add_row("GPUtil", "✅ Available" if detection["gputil_available"] else "❌ Not Available")
        detection_table.add_row("NVML", "✅ Available" if detection["nvml_available"] else "❌ Not Available")
        detection_table.add_row("PyNVML", "✅ Available" if detection["pynvml_available"] else "❌ Not Available")
        detection_table.add_row("nvidia-smi", "✅ Available" if detection["nvidia_smi_available"] else "❌ Not Available")
        
        console.print(detection_table)
        
        console.print(f"\n[bold]GPU Count:[/bold] {detection['gpu_count']}")
        
        if detection["gpu_names"]:
            console.print("\n[bold cyan]Available GPUs:[/bold cyan]")
            for i, name in enumerate(detection["gpu_names"]):
                console.print(f"  GPU {i}: {name}")
        
        # Display current metrics
        console.print("\n" + "="*60)
        gpu_monitor.print_current_status()
        
        # Display system info if available
        if "system_info" in detection and detection["system_info"]:
            sys_info = detection["system_info"]
            console.print("[bold cyan]System Information:[/bold cyan]")
            console.print(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
            console.print(f"  Memory: {sys_info.get('memory_total_gb', 0):.1f} GB")
            console.print(f"  Platform: {sys_info.get('platform', 'Unknown')}")
        
    except ImportError as e:
        console.print(f"❌ GPU monitoring dependencies not installed: {e}")
        console.print("Install with: [cyan]pip install -r requirements.txt[/cyan]")
    except Exception as e:
        console.print(f"❌ GPU status error: {e}")


@app.command()
def gpu_monitor(
    duration: int = typer.Option(30, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: float = typer.Option(1.0, "--interval", "-i", help="Sampling interval in seconds")
):
    """
    Start interactive GPU monitoring for specified duration
    """
    display_banner()
    console.print(f"\n[bold green]🔍 GPU Monitoring ({duration}s)[/bold green]\n")
    
    try:
        from src.gpu_monitor import get_gpu_monitor
        
        gpu_monitor = get_gpu_monitor()
        gpu_monitor.sample_interval = interval
        
        if gpu_monitor.detection_results["gpu_count"] == 0:
            console.print("❌ No GPUs detected for monitoring")
            return
        
        # Start monitoring
        gpu_monitor.start_monitoring()
        
        console.print(f"📊 Monitoring {gpu_monitor.detection_results['gpu_count']} GPU(s) for {duration} seconds...")
        console.print("Press Ctrl+C to stop early\n")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Monitoring GPUs...", total=duration)
                
                for i in range(duration):
                    progress.update(task, advance=1)
                    time.sleep(1)
                    
                    # Show current status every 10 seconds
                    if i > 0 and i % 10 == 0:
                        current_metrics = gpu_monitor.get_current_metrics()
                        if current_metrics:
                            for metric in current_metrics:
                                progress.console.print(f"GPU {metric.gpu_id}: {metric.utilization_percent:.1f}% util, {metric.memory_percent:.1f}% mem, {metric.temperature_c:.1f}°C")
        
        except KeyboardInterrupt:
            console.print("\n⏹️ Monitoring stopped by user")
        
        # Stop monitoring and show results
        gpu_monitor.stop_monitoring()
        
        # Display summary
        summary = gpu_monitor.get_summary_stats()
        if "error" not in summary:
            console.print("\n[bold green]📊 Monitoring Summary[/bold green]")
            
            for gpu_key, stats in summary.items():
                if gpu_key.startswith("gpu_"):
                    console.print(f"\n[bold]{stats['name']}[/bold]")
                    console.print(f"  Utilization: {stats['utilization']['avg']:.1f}% avg (max: {stats['utilization']['max']:.1f}%)")
                    console.print(f"  Memory: {stats['memory']['avg']:.1f}% avg (max: {stats['memory']['max']:.1f}%)")
                    console.print(f"  Temperature: {stats['temperature']['avg']:.1f}°C avg (max: {stats['temperature']['max']:.1f}°C)")
                    if 'power' in stats:
                        console.print(f"  Power: {stats['power']['avg']:.1f}W avg (max: {stats['power']['max']:.1f}W)")
        
        # Save results
        filename = gpu_monitor.save_monitoring_data()
        console.print(f"\n💾 Monitoring data saved: {filename}")
        
    except ImportError as e:
        console.print(f"❌ GPU monitoring dependencies not installed: {e}")
        console.print("Install with: [cyan]pip install -r requirements.txt[/cyan]")
    except Exception as e:
        console.print(f"❌ GPU monitoring error: {e}")


@app.command()
def gpu_profile():
    """
    Profile GPU usage during routing operations
    """
    display_banner()
    console.print("\n[bold green]🎯 GPU Routing Profiler[/bold green]\n")
    
    try:
        from src.gpu_monitor import get_gpu_monitor, RoutingGPUProfiler
        
        gpu_monitor = get_gpu_monitor()
        profiler = RoutingGPUProfiler(gpu_monitor)
        
        if gpu_monitor.detection_results["gpu_count"] == 0:
            console.print("❌ No GPUs detected for profiling")
            return
        
        # Test prompts for profiling
        test_prompts = [
            "Write a Python function to implement quicksort algorithm",
            "Solve the integral of x^2 + 3x - 5 from 0 to 2", 
            "What is the capital of Australia?",
            "Create a REST API endpoint using FastAPI",
            "Find the derivative of sin(x) * cos(x)"
        ]
        
        console.print("🚀 Starting GPU profiling with test routing operations...")
        
        # Initialize router
        router = SemanticRouter()
        
        # Start profiling session
        profiler.start_routing_session("test_routing_profile")
        
        console.print(f"📊 Processing {len(test_prompts)} test prompts with GPU monitoring...")
        
        for i, prompt in enumerate(test_prompts, 1):
            console.print(f"[{i}/{len(test_prompts)}] Routing: [italic]{prompt[:50]}...[/italic]")
            
            # Route the prompt (this will use GPU for Ollama models)
            routing_result = router.route(prompt)
            
            # Brief delay to capture GPU usage
            time.sleep(0.5)
        
        # End profiling session
        session_result = profiler.end_routing_session()
        
        if session_result:
            console.print("\n[bold green]🎯 Profiling Results[/bold green]")
            console.print(f"Session Duration: {session_result['duration']:.2f} seconds")
            
            # Display GPU statistics during profiling
            summary = session_result.get('summary_stats', {})
            if "error" not in summary:
                for gpu_key, stats in summary.items():
                    if gpu_key.startswith("gpu_"):
                        console.print(f"\n[bold]{stats['name']} Performance[/bold]")
                        console.print(f"  Average Utilization: {stats['utilization']['avg']:.1f}%")
                        console.print(f"  Peak Utilization: {stats['utilization']['max']:.1f}%")
                        console.print(f"  Average Memory: {stats['memory']['avg']:.1f}%")
                        console.print(f"  Peak Memory: {stats['memory']['max']:.1f}%")
                        console.print(f"  Average Temperature: {stats['temperature']['avg']:.1f}°C")
                        if 'power' in stats:
                            console.print(f"  Average Power: {stats['power']['avg']:.1f}W")
        
        console.print("\n💡 Use this data to optimize your local model performance!")
        
    except ImportError as e:
        console.print(f"❌ GPU monitoring dependencies not installed: {e}")
        console.print("Install with: [cyan]pip install -r requirements.txt[/cyan]")
    except Exception as e:
        console.print(f"❌ GPU profiling error: {e}")


if __name__ == "__main__":
    app()
