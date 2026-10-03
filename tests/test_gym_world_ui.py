"""Live-world configuration and selected-agent controls in the browser."""

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
    page.wait_for_function("document.querySelector('gym-view')?.catalogue.length > 0")
    return page, errors


def test_world_modes_manual_validation_and_online_checkpoint(gym_page):
    page, errors = gym_page
    page.get_by_label('Simulation mode', exact=True).select_option('world')
    assert page.get_by_label('Controller', exact=True).input_value() == 'native-idm'
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
