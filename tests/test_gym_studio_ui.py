"""Scenario selection and studio layout in the browser."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
pytestmark = pytest.mark.slow


def test_scenario_cards_drive_setup_and_preserve_live_control_access(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("document.querySelector('gym-view')?.catalogue.length > 0")
    page.locator('gym-view .gym-scenario[data-environment="warehouse"] button').first.click()
    assert page.get_by_label('Environment', exact=True).input_value() == 'warehouse'
    assert page.get_by_label('Controller', exact=True).input_value() == 'manual'
    assert page.locator('gym-view .gym-scenario[data-environment="warehouse"] button').first.get_attribute('aria-pressed') == 'true'
    page.locator('gym-view .gym-scenario[data-environment="car"] button').first.click()
    assert page.get_by_label('Environment', exact=True).input_value() == 'car'
    assert page.get_by_label('Controller', exact=True).input_value() == 'decider'
    assert not page.locator('gym-view #config .gym-advanced').get_attribute('open')
    assert not page.get_by_label('Seed', exact=True).is_visible()
    assert not page.locator('gym-view .gym-world-settings').get_attribute('open')
    start = page.get_by_role('button', name='Start session', exact=True).bounding_box()
    assert start and start['y'] + start['height'] < page.viewport_size['height']
    assert page.locator('gym-view .gym-stage').is_visible()
    assert page.get_by_role('button', name='Single step', exact=True).is_visible()
    assert not page.get_by_role('button', name='Apply manual action', exact=True).is_visible()
    assert not page.get_by_role('button', name='Refresh recordings', exact=True).is_visible()
    page.locator('gym-view .gym-recordings-panel > summary').click()
    assert page.get_by_role('button', name='Refresh recordings', exact=True).is_visible()
    page.locator('gym-view .gym-recordings-panel > summary').click()
    layout = page.evaluate("""() => {
      const g=document.querySelector('gym-view'),setup=g.querySelector('#config').getBoundingClientRect(),
        stage=g.querySelector('.gym-stage').getBoundingClientRect();
      return {setup:setup.width,stage:stage.width,canvasRight:stage.left>=setup.right};
    }""")
    assert layout['canvasRight'] and layout['stage'] > layout['setup']
    page.locator('gym-view #config .gym-advanced > summary').click()
    assert page.get_by_label('Seed', exact=True).is_visible()
    assert not errors


@pytest.mark.parametrize('theme', ['dark', 'light', 'custom'])
def test_rl_text_and_brand_surfaces_remain_readable_across_themes(joined, open_page, theme):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("document.querySelector('gym-view')?.catalogue.length > 0")
    page.locator('gym-view .gym-scenario[data-environment="car"] button').first.click()
    result = page.evaluate("""theme => {
      const root=document.documentElement;
      root.dataset.theme=theme==='custom'?'dark':theme;
      if(theme==='custom')for(const [key,value]of Object.entries({
        '--ml-bg':'#102f28','--ml-surface':'#183d34','--ml-sunken':'#123329',
        '--ml-text':'#f3fff4','--ml-muted':'#b7d4bf','--ml-accent':'#f6c865',
        '--ml-on-accent':'#253522'}))root.style.setProperty(key,value);
      const style=node=>getComputedStyle(node);
      const luminance=color=>{
        const channels=color.match(/[\d.]+/g).slice(0,3).map(Number).map(v=>{
          v/=255;return v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4);
        });
        return channels[0]*.2126+channels[1]*.7152+channels[2]*.0722;
      };
      const contrast=(foreground,background)=>{
        const values=[luminance(foreground),luminance(background)].sort((a,b)=>b-a);
        return (values[0]+.05)/(values[1]+.05);
      };
      const gym=document.querySelector('gym-view'),ready=gym.querySelector('.gym-ready');
      const checks=[];
      for(const selector of ['.gym-ready','.gym-scenario h3','.gym-scenario p:not(.gym-ready)']){
        const node=gym.querySelector(selector),card=node.closest('.gym-scenario');
        checks.push({selector,ratio:contrast(style(node).color,style(card).backgroundColor)});
      }
      for(const selector of ['.gym-start','.gym-scenario[data-selected=true] button','#controls button:first-child','.gym-scenario-icon']){
        const node=gym.querySelector(selector);
        checks.push({selector,ratio:contrast(style(node).color,style(node).backgroundColor)});
      }
      const stage=gym.querySelector('.gym-stage');
      for(const selector of ['#scene-empty strong','#scene-empty span']){
        const node=gym.querySelector(selector);
        checks.push({selector,ratio:contrast(style(node).color,style(stage).backgroundColor)});
      }
      return {checks,ready:style(ready).color,text:style(gym.querySelector('.workspace')).color};
    }""", theme)
    assert result['ready'] == result['text']
    for check in result['checks']:
        assert check['ratio'] >= 4.5, (theme, check)
    assert not errors
