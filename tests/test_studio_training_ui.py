"""Guided Studio training controls driven in the browser."""

import pytest
import test_fleet_page as fleet_page
from playwright.sync_api import expect

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


def test_training_steps_validate_dataset_and_preserve_recipe_arguments(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    expect(page.get_by_label('Dataset path (relative to files root)')).to_be_visible()
    page.get_by_role('button', name='Continue to model & recipe').click()
    expect(page.locator('training-view #config > .status')).to_contain_text('Choose a dataset')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/demo.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    page.get_by_label('Base model ID or local directory').fill('models/demo-base')
    page.get_by_label('Fine-tuning strategy').select_option('adapter')
    page.get_by_role('button', name='Review this run').click()
    expect(page.get_by_role('button', name='Queue 20-step training smoke', exact=True)).to_be_visible()
    page.get_by_role('button', name='Review command', exact=True).click()
    expect(page.locator('training-view #config > .status')).to_contain_text('trains for 20 steps')
    spec = page.evaluate("document.querySelector('training-view').spec()")
    from ml_stack.train.run import _parser
    parsed = _parser().parse_args(spec['args'])
    assert parsed.recipe == 'tool-calls' and parsed.lora
    assert parsed.data == 'datasets/demo.jsonl' and 'base=models/demo-base' in parsed.set
    page.get_by_role('button', name='Back', exact=True).click()
    assert page.get_by_label('Base model ID or local directory').input_value() == 'models/demo-base'
    page.get_by_label('Fine-tuning strategy').select_option('full')
    parsed = _parser().parse_args(page.evaluate("document.querySelector('training-view').spec()")['args'])
    assert not parsed.lora and 'lora=false' in parsed.set
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors


def test_dataset_handoff_opens_model_step_and_keeps_run_summary(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.evaluate("sessionStorage.setItem('ml-stack-dataset','datasets/chosen.jsonl')")
    page.evaluate("window.fleetModel.go('models'); window.fleetModel.go('training')")
    expect(page.get_by_label('Recipe', exact=True)).to_be_visible()
    expect(page.locator('training-view #run-summary')).to_contain_text('datasets/chosen.jsonl')
    page.get_by_role('button', name='Back', exact=True).click()
    assert page.get_by_label('Dataset path (relative to files root)').input_value() == 'datasets/chosen.jsonl'
    assert not errors


@pytest.mark.parametrize('model_path', ['/tmp/exact-model.gguf', r'C:\models\exact-model.gguf'])
def test_library_chat_action_selects_server_identity(joined, open_page, model_path):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#models')
    row = {'id': 'display-id', 'name': 'Friendly model', 'path': model_path,
           'family': 'Qwen', 'format': 'gguf', 'quantization': 'Q4', 'kind': 'text',
           'status': 'installed', 'servable': True, 'size_bytes': 1000,
           'shards': 1, 'is_complete': True, 'files': []}
    page.route('**/ui/models', lambda route: route.fulfill(json={
        'library': [row], 'here': [], 'elsewhere': [], 'getting': [], 'unfinished': [],
    }))
    page.route('**/ui/serving', lambda route: route.fulfill(json={
        'running': [{'models': ['exact-model.gguf'], 'port': 12345}], 'can_serve': True,
    }))
    page.route('**/ui/chat', lambda route: route.fulfill(json={
        'models': [{'model': 'exact-model.gguf', 'local': True}], 'runtime_ready': True,
    }))
    page.evaluate("document.querySelector('models-view').draw()")
    page.locator('models-library').get_by_role('button', name='Open chat', exact=True).click()
    expect(page).to_have_url(f'http://127.0.0.1:{joined.port}/ui/#chat')
    expect(page.locator('chat-view #model')).to_have_value('exact-model.gguf')
    assert not errors


def test_decision_tools_handoff_opens_recipe_step(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#tools')
    page.wait_for_function("() => window.fleetModel.route === 'tools'")
    page.evaluate("document.querySelector('training-view').openRun({workflow:'decider',dataset:'datasets/decisions.jsonl'})")
    expect(page.get_by_label('Workflow', exact=True)).to_have_value('decider')
    expect(page.get_by_label('Recipe', exact=True)).to_be_visible()
    page.get_by_role('button', name='Review this run').click()
    spec = page.evaluate("document.querySelector('training-view').spec()")
    assert spec['command'] == 'ml-stack-decide'
    assert spec['args'][spec['args'].index('--data') + 1] == 'datasets/decisions.jsonl'
    assert not errors


def test_review_reveals_invalid_advanced_training_field(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    expect(page.get_by_label('Workflow', exact=True)).to_be_visible()
    page.get_by_label('Workflow', exact=True).select_option('decider')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/decisions.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    advanced = page.locator('training-view #training-recipe details[data-advanced]')
    advanced.locator('summary').click()
    page.get_by_label('Batch size', exact=True).fill('0')
    advanced.locator('summary').click()
    page.get_by_role('button', name='Review this run').click()
    posts = []
    page.on('request', lambda request: posts.append(request.url)
            if request.method == 'POST' and request.url.endswith('/ui/workspace/jobs') else None)
    page.get_by_role('button', name='Review command', exact=True).click()
    expect(page.get_by_label('Batch size', exact=True)).to_be_visible()
    expect(page.get_by_label('Batch size', exact=True)).to_be_focused()
    expect(page.locator('training-view #config > .status')).to_contain_text('highlighted training option')
    assert not posts
    page.get_by_label('Batch size', exact=True).fill('1')
    page.get_by_role('button', name='Review this run').click()
    page.get_by_role('button', name='Review command', exact=True).click()
    expect(page.locator('training-view #config > .status')).to_contain_text('Command preview ready')
    assert len(posts) == 1
    assert not errors


def test_fresh_rl_workflow_requires_no_dataset(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    expect(page.get_by_label('Workflow', exact=True)).to_be_visible()
    page.get_by_label('Workflow', exact=True).select_option('rl')
    expect(page.locator('training-view').get_by_label('Environment', exact=True)).to_be_visible()
    expect(page.get_by_label('Training timesteps', exact=True)).to_be_visible()
    assert page.get_by_label('Dataset path (relative to files root)').input_value() == ''
    assert not page.get_by_label('Dataset path (relative to files root)').is_visible()
    spec = page.evaluate("document.querySelector('training-view').spec()")
    assert spec['command'] == 'ml-stack-gym' and '--data' not in spec['args']
    assert not errors


@pytest.mark.parametrize('theme', [
    {'--bg-raised': '#ffffff', '--text': '#1b1f3a', '--good': '#14816c'},
    {'--bg-raised': '#1a1d28', '--text': '#f4f4ff', '--good': '#68dfbc'},
    {'--bg-raised': '#29222e', '--text': '#fff2dc', '--good': '#ffd166'},
])
def test_running_model_badge_preserves_theme_contrast(joined, open_page, theme):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#models')
    page.wait_for_function("() => window.fleetModel.route === 'models'")
    page.evaluate("""theme => {
      for (const [key,value] of Object.entries(theme)) document.documentElement.style.setProperty(key,value);
      const library=document.querySelector('models-library');
      library.update([{id:'theme-model',name:'Theme model',path:'/tmp/theme.gguf',family:'Qwen',
        format:'gguf',quantization:'Q4',kind:'text',status:'installed',servable:true,size_bytes:1,
        shards:1,is_complete:true,files:[]}],{can_serve:true,running:[{models:['theme.gguf'],port:12345}]},()=>{},()=>{});
    }""", theme)
    contrast = page.locator('models-library .model-running').evaluate("""node => {
      const style=getComputedStyle(node),canvas=document.createElement('canvas');canvas.width=canvas.height=1;
      const context=canvas.getContext('2d');
      const luminance=css=>{context.clearRect(0,0,1,1);context.fillStyle=css;context.fillRect(0,0,1,1);
        const data=context.getImageData(0,0,1,1).data;
        const values=[...data].slice(0,3).map(channel=>{channel/=255;return channel<=.04045?channel/12.92:((channel+.055)/1.055)**2.4;});
        return .2126*values[0]+.7152*values[1]+.0722*values[2];};
      const text=luminance(style.color),surface=luminance(style.backgroundColor);
      return (Math.max(text,surface)+.05)/(Math.min(text,surface)+.05);
    }""")
    assert contrast >= 4.5
    assert not errors


def test_changing_workflow_invalidates_review_and_explains_missing_runtime(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/examples.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_role('button', name='Review this run').click()
    page.get_by_role('button', name='Review command', exact=True).click()
    expect(page.locator('training-view #config > .status')).to_contain_text('Command preview ready')
    page.evaluate("document.querySelector('training-view').environments.forEach(environment=>{environment.available=false;environment.missing=['test-runtime'];})")
    page.get_by_label('Workflow', exact=True).select_option('rl')
    expect(page.locator('training-view #config > .status')).to_be_empty()
    assert page.locator('training-view').get_by_text('Command preview', exact=True).locator('..').get_attribute('open') is None
    page.get_by_role('button', name='Review this run').click()
    expect(page.get_by_role('button', name='Review command', exact=True)).to_be_disabled()
    expect(page.locator('#training-runtime')).to_be_visible()
    expect(page.locator('#training-runtime')).to_contain_text('test-runtime')
    assert not errors


def test_review_response_does_not_restore_command_after_options_change(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/examples.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_role('button', name='Review this run').click()
    deferred = []
    page.route('**/ui/workspace/jobs', lambda route: deferred.append(route)
               if route.request.method == 'POST' else route.continue_())
    page.get_by_role('button', name='Review command', exact=True).click()
    page.wait_for_timeout(100)
    assert len(deferred) == 1
    page.get_by_label('Run name', exact=True).fill('changed-run')
    deferred[0].fulfill(json={'command': 'obsolete preview'})
    expect(page.locator('training-view #config > .status')).to_contain_text('Options changed')
    assert not page.locator('training-view').get_by_text('obsolete preview', exact=True).count()
    assert not errors


def test_mlx_backend_choice_survives_recipe_rebuild_and_sets_payload(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/demo.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    page.get_by_label('Model size', exact=True).select_option('qwen27b')
    expect(page.get_by_label('Training backend')).to_have_value('mlx')
    page.get_by_label('Training backend').select_option('torch')
    page.get_by_label('Model size', exact=True).select_option('e4b')
    page.get_by_label('Model size', exact=True).select_option('qwen27b')
    expect(page.get_by_label('Training backend')).to_have_value('torch')
    page.get_by_label('Training backend').select_option('mlx')
    page.get_by_role('button', name='Review this run').click()
    spec = page.evaluate("document.querySelector('training-view').spec()")
    assert 'framework=mlx' in spec['args'] and 'lora=true' in spec['args']
    assert 'steps=20' in spec['args'] and 'context=512' in spec['args']
    assert not errors


def test_missing_mlx_training_library_explains_disabled_start(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.route('**/ui/libraries', lambda route: route.fulfill(json={
        'libraries': [{'name': 'train-mlx', 'title': 'MLX language model fine-tuning', 'installed': False}]}))
    page.evaluate("document.querySelector('training-view').load()")
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/demo.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    page.get_by_label('Model size', exact=True).select_option('qwen27b')
    page.get_by_role('button', name='Review this run').click()
    expect(page.locator('training-view #training-runtime')).to_contain_text('Install MLX language model fine-tuning in Settings')
    expect(page.get_by_role('button', name='Queue 20-step training smoke', exact=True)).to_be_disabled()
    expect(page.get_by_role('button', name='Review command', exact=True)).to_be_enabled()
    assert not errors


@pytest.mark.parametrize('inventory', ['absent', 'failed'])
def test_unavailable_mlx_inventory_blocks_queue_and_retains_review(joined, open_page, inventory):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#training')
    page.route('**/ui/libraries', lambda route: route.fulfill(
        status=503 if inventory == 'failed' else 200,
        json={'error': 'inventory unavailable'} if inventory == 'failed' else {'libraries': []}))
    page.evaluate("document.querySelector('training-view').load()")
    page.get_by_label('Dataset path (relative to files root)').fill('datasets/demo.jsonl')
    page.get_by_role('button', name='Continue to model & recipe').click()
    page.get_by_label('Recipe', exact=True).select_option('tool-calls')
    page.get_by_label('Model size', exact=True).select_option('qwen27b')
    page.get_by_role('button', name='Review this run').click()
    expect(page.get_by_role('button', name='Queue 20-step training smoke', exact=True)).to_be_disabled()
    expect(page.get_by_role('button', name='Review command', exact=True)).to_be_enabled()
    expect(page.locator('training-view #training-runtime')).to_contain_text(
        'Could not verify' if inventory == 'failed' else 'not available on this device')
    assert not errors
