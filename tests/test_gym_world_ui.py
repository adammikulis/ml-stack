"""Live-world configuration and selected-agent controls in the browser."""

import json

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


@pytest.fixture
def gym_page(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("() => document.querySelector('gym-view')?.catalogue.length > 0")
    page.get_by_label('Environment', exact=True).select_option('car')
    page.locator('gym-view .gym-world-settings > summary').click()
    page.get_by_label('Simulation mode', exact=True).select_option('world')
    return page, errors


def test_world_modes_manual_validation_and_online_checkpoint(gym_page):
    page, errors = gym_page
    page.get_by_label('Simulation mode', exact=True).select_option('world')
    assert page.get_by_label('Controller', exact=True).input_value() == 'decider'
    page.get_by_label('World source', exact=True).select_option('manual')
    page.get_by_role('button', name='Start session', exact=True).click()
    assert 'manual world files or layout' in page.locator('#gym-note').inner_text()
    page.get_by_label('Map file', exact=True).fill('worlds/country.json')
    page.get_by_label('Controller', exact=True).select_option('ppo')
    page.get_by_label('Policy learning', exact=True).select_option('online')
    config = page.evaluate("""() => {const g=document.querySelector('gym-view');
        g.worldOptions.validate();return g.worldOptions.value();}""")
    assert config == {'simulation_mode': 'world', 'learning_mode': 'online',
                      'world': {'mode': 'manual', 'map_file': 'worlds/country.json'}}
    page.get_by_label('Policy learning', exact=True).select_option('frozen')
    page.get_by_role('button', name='Start session', exact=True).click()
    assert 'Choose a PPO checkpoint' in page.locator('#gym-note').inner_text()
    page.get_by_label('PPO checkpoint path').fill('/tmp/policy.zip')
    page.get_by_label('Simulation mode', exact=True).select_option('episode')
    assert not page.get_by_label('Policy learning', exact=True).is_visible()
    assert page.evaluate("document.querySelector('gym-view').worldOptions.value().learning_mode") == 'frozen'
    assert not errors


def test_drone_online_learning_is_explicit_ppo_and_freeze_keeps_session(gym_page):
    page, errors = gym_page
    page.get_by_label('Environment', exact=True).select_option('drone')
    page.get_by_label('Controller', exact=True).select_option('native-patrol')
    assert page.get_by_label('Policy learning', exact=True).input_value() == 'frozen'
    assert page.locator('gym-world-options option[value="online"]').is_disabled()
    page.get_by_label('Controller', exact=True).select_option('ppo')
    page.get_by_label('Policy learning', exact=True).select_option('online')
    result = page.evaluate("""async () => {
      const g=document.querySelector('gym-view'),calls=[];g.session='persistent-drone';
      g.control=async(command,payload)=>calls.push({command,payload});
      const online=g.worldOptions.value();g.worldOptions.learning.value='frozen';
      g.checkpoint.value='/tmp/drone-policy.zip';await g.worldOptions.applyLearning();
      const session=g.session;g.session=null;return {online,calls,session};
    }""")
    assert result['online']['learning_mode'] == 'online'
    assert result['calls'] == [{'command': 'learning', 'payload': {'mode': 'frozen'}}]
    assert result['session'] == 'persistent-drone' and not errors


def test_selected_actor_has_own_telemetry_and_explicit_control_handoff(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {
      const g=document.querySelector('gym-view');g.session='live';g.selectedAgent='traffic';g.selectedKind='vehicle';
      const calls=[];g.control=async(command,payload)=>calls.push({command,payload});
      g.draw({environment:'car',controller:'ppo',sequence:9,reward:999,action:8,
        decision:{choice:'EGO_ONLY',probabilities:{EGO_ONLY:.99}},info:{render:{vehicle:{id:'ego'},
          vehicles:[{id:'ego',ego:true},{id:'traffic',position:[3,4],speed:8,controller:'IDM',last_action:[.2,.5]}]}}},false);
      g.querySelector('gym-scene').selectAgent('vehicle-traffic');
      const text=g.querySelector('#decision-flow').innerText;
      const before=calls.length;await g.takeControl();g.session=null;return {text,before,calls};
    }""")
    assert 'EGO_ONLY' not in result['text'] and '999' not in result['text']
    assert 'Speed 8.0 m/s' in result['text'] and 'IDM' in result['text']
    assert result['before'] == 0
    assert result['calls'] == [{'command': 'agent', 'payload': {'agent_id': 'traffic'}}]
    assert not errors


def test_learning_switch_preserves_world_and_restores_settings(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {
      const g=document.querySelector('gym-view');g.session='live';g.controller.value='ppo';g.checkpoint.value='/tmp/policy.zip';
      g.worldOptions.restore({simulation_mode:'world',learning_mode:'online',world:{mode:'manual',map_file:'maps/saved.json'}});
      const restored=g.worldOptions.value(),calls=[];g.control=async(command,payload)=>calls.push({command,payload});
      g.worldOptions.learning.value='frozen';await g.worldOptions.applyLearning();g.session=null;
      return {restored,calls};
    }""")
    assert result['restored']['world']['map_file'] == 'maps/saved.json'
    assert result['restored']['learning_mode'] == 'online'
    assert result['calls'] == [{'command': 'learning', 'payload': {'mode': 'frozen'}}]
    assert not errors


def test_applying_online_ppo_creates_policy_without_resetting_world(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {
      const g=document.querySelector('gym-view');g.session='persistent';g.controller.value='ppo';
      g.worldOptions.learning.value='online';g.checkpoint.value='';
      const calls=[];g.control=async(command,payload)=>calls.push({command,payload});
      await g.apply();g.session=null;return calls;
    }""")
    assert len(result) == 1 and result[0]['command'] == 'controller'
    assert result[0]['payload']['learning_mode'] == 'online'
    assert result[0]['payload']['checkpoint'] == ''
    assert not errors


@pytest.mark.parametrize('layout', ['S', 'SC', '3'])
def test_road_layout_is_the_only_map_source_in_start_request(gym_page, layout):
    page, errors = gym_page
    page.route('**/ui/gym/sessions', lambda route: route.fulfill(
        status=200, content_type='application/json',
        body=json.dumps({'ok': True, 'id': 'map-config-proof', 'status': 'paused',
                         'environment': 'car', 'sequence': 0})))
    page.evaluate("document.querySelector('gym-view').stream=async()=>{}")
    page.get_by_label('Road layout', exact=True).select_option(layout)
    assert page.get_by_label('map', exact=True).count() == 0
    with page.expect_request(lambda request: request.method == 'POST' and request.url.endswith('/ui/gym/sessions')) as sent:
        page.get_by_role('button', name='Start session', exact=True).click()
    config = sent.value.post_data_json['config']
    assert config['world']['map'] == (3 if layout == '3' else layout)
    assert 'map' not in config
    assert page.evaluate("""() => {const gym=document.querySelector('gym-view');
      return gym.worldOptions.fields.map.input === gym.nativeFields.map;}""")
    page.evaluate("document.querySelector('gym-view').session=null")
    assert not errors


def test_manual_source_hides_road_layout_and_warehouse_has_no_registration_override(gym_page):
    page, errors = gym_page
    page.get_by_label('World source', exact=True).select_option('manual')
    assert not page.get_by_label('Road layout', exact=True).is_visible()
    page.get_by_label('World source', exact=True).select_option('procedural')
    page.get_by_label('Road layout', exact=True).select_option('SC')
    page.get_by_label('Environment', exact=True).select_option('warehouse')
    assert page.get_by_label('Warehouse scenario', exact=True).count() == 0
    assert 'env_id' not in page.evaluate("document.querySelector('gym-view').nativeConfig()")
    page.get_by_label('Environment', exact=True).select_option('car')
    assert page.get_by_label('Road layout', exact=True).input_value() == 'SC'
    assert not errors


def test_attach_restores_map_from_world_definition(gym_page):
    page, errors = gym_page
    snapshot = {'ok': True, 'id': 'attached-map', 'environment': 'car', 'controller': 'native-idm',
                'sequence': 0, 'status': 'paused', 'config': {
                    'simulation_mode': 'world', 'map': 'SCSCS',
                    'world': {'mode': 'procedural', 'map': 'SC'}}}
    page.route('**/ui/gym/sessions/attached-map', lambda route: route.fulfill(
        status=200, content_type='application/json', body=json.dumps(snapshot)))
    page.evaluate("""() => {const g=document.querySelector('gym-view');g.stream=async()=>{};
        const option=document.createElement('option');option.value='attached-map';
        option.textContent='Saved country road';g.live.append(option);}""")
    page.locator('gym-view #config > .gym-advanced > summary').click()
    page.get_by_label('Live sessions', exact=True).select_option('attached-map')
    page.get_by_role('button', name='Attach to live session', exact=True).click()
    page.wait_for_function("""() => {const g=document.querySelector('gym-view');
        return g.session === 'attached-map' && g.nativeFields.map.value === 'SC';}""")
    assert page.get_by_label('Road layout', exact=True).input_value() == 'SC'
    assert page.evaluate("document.querySelector('gym-view').worldOptions.value().world.map") == 'SC'
    assert 'map' not in json.loads(page.get_by_label('Additional native settings (JSON)').input_value())
    page.evaluate("document.querySelector('gym-view').session=null")
    assert not errors


def test_models_apply_to_live_world_without_reset_or_extra_click(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {
      const g=document.querySelector('gym-view');g.session='persistent';const calls=[];
      g.control=async(command,payload)=>calls.push({command,payload});
      g.modelOptions.choices({decision:[{id:'strands',label:'Strands Decider 2B (default)',checkpoint:null},
        {id:'trained',label:'Country policy',checkpoint:'/models/trained'}],vision:[]});
      g.controller.value='decider';g.modelOptions.decision.value='trained';
      g.modelOptions.decision.dispatchEvent(new Event('change'));await new Promise(resolve=>setTimeout(resolve,10));
      g.controller.value='native-idm';g.controller.dispatchEvent(new Event('change'));
      await new Promise(resolve=>setTimeout(resolve,10));g.session=null;return calls;
    }""")
    assert [call['command'] for call in result] == ['controller', 'controller']
    assert result[0]['payload']['decision_checkpoint'] == '/models/trained'
    assert result[1]['payload']['controller'] == 'native-idm'
    assert not errors


def test_setup_initializes_world_fields_and_single_step_keeps_policy(gym_page):
    page, errors = gym_page
    assert page.get_by_label('Road layout', exact=True).input_value() == 'SCSCS'
    page.evaluate("""() => {const g=document.querySelector('gym-view');
      g.session='step-policy';g.controller.value='ppo';g.checkpoint.value='/tmp/policy.zip';
      g.calls=[];g.control=async(command,payload)=>g.calls.push({command,payload});g.paintActions();}""")
    page.get_by_role('button', name='Single step', exact=True).click()
    assert page.evaluate("document.querySelector('gym-view').calls.map(call=>call.command)") == ['step']
    page.evaluate("document.querySelector('gym-view').session=null")
    assert not errors


def test_invalid_json_preserves_attached_session(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {const g=document.querySelector('gym-view');
      g.session='preserved-world';g.config.value='[]';const closed=[];g.close=async()=>closed.push(g.session);
      await g.start();const result={closed,session:g.session,note:g.note.textContent};g.session=null;return result;}""")
    assert result['closed'] == [] and result['session'] == 'preserved-world'
    assert 'JSON object' in result['note']
    assert not errors


def test_repeated_start_submits_one_session_and_preserves_existing_world(gym_page):
    page, errors = gym_page
    result = page.evaluate("""async () => {const g=document.querySelector('gym-view'),w=window.workspaceModel;
      g.session='original-world';g.stream=async()=>{};g.loadSessions=async()=>{};
      const original=w.post,calls=[];let finish;w.post=async(path,body)=>{calls.push({path,body});return new Promise(resolve=>finish=resolve);};
      const first=g.start();await g.start();const before=g.session;
      finish({id:'new-world',environment:'car',controller:'decider',status:'paused',sequence:0});await first;
      w.post=original;const result={calls,before,after:g.session,pending:g.pending};g.session=null;return result;}""")
    assert len(result['calls']) == 1
    assert result['before'] == 'original-world' and result['after'] == 'new-world'
    assert not result['pending'] and not errors
