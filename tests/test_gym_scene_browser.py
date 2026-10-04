"""Native geometry, sensor debug and camera controls in the Three renderer."""

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page

pytestmark = pytest.mark.slow


@pytest.fixture
def scene_page(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie, path='/ui/#gym')
    page.wait_for_function("document.querySelector('gym-scene')?.renderer !== undefined")
    page.evaluate("""() => {
        const scene = document.querySelector('gym-scene');
        scene.hidden = false;
        scene.update({environment:'car', heading_unit:'radians',
          roads:[{id:'native-curve',width:3.5,points:[[0,0],[15,0],[30,4],[40,15],[42,35]]}],
          vehicle:{id:'ego'}, vehicles:[{id:'ego',position:[8,0],heading:0,
            length:4.6,width:1.9,height:1.6,last_action:[.25,.7]}],
          lidar:{origin:[8,0],endpoints:[[18,0],[8,30]],distances_m:[10,30],range_m:30},
          stop_signs:[{id:'route-stop',position:[30,7],heading:0,heading_unit:'radians',stop_line:[[28,2],[28,5]]}]});
    }""")
    return page, errors


def test_sensor_endpoints_hits_native_heading_and_physical_materials(scene_page):
    page, errors = scene_page
    result = page.evaluate("""() => {
        const s=document.querySelector('gym-scene'), vehicle=s.meshes.get('vehicle-ego');
        const rays=s.meshes.get('sensor-rays').geometry;
        return {rays:Number(s.dataset.rayCount), hits:Number(s.dataset.hitCount),
          endpoints:Array.from(rays.getAttribute('position').array),
          heading:vehicle.userData.heading, target:vehicle.userData.target.toArray(),
          paint:vehicle.children[0].material.type,clearcoat:vehicle.children[0].material.clearcoat,
          scale:vehicle.scale.toArray(),control:s.meshes.has('control-ego'),
          stop:s.meshes.has('stop-route-stop'),line:s.meshes.has('stop-line-route-stop')};
    }""")
    assert result['rays'] == 2 and result['hits'] == 1
    assert result['endpoints'] == pytest.approx([8,.65,0,18,.65,0,8,.65,0,8,.65,30])
    assert result['heading'] == pytest.approx(1.5707963267948966)
    assert result['target'] == [8,0,0]
    assert result['paint'] == 'MeshPhysicalMaterial' and result['clearcoat'] == 1
    assert result['scale'] == pytest.approx([1.9/1.8,1.6/1.51,4.6/4.5])
    assert result['control'] and result['stop'] and result['line']
    assert not errors


def test_follow_orbit_sensor_toggle_and_whole_map_controls(scene_page):
    page, errors = scene_page
    page.wait_for_function("""() => {
      const s=document.querySelector('gym-scene');
      return s.camera.position.distanceTo(s.meshes.get('vehicle-ego').position)<25;
    }""")
    scene = page.locator('gym-scene')
    scene.get_by_label('Sensor rays', exact=True).uncheck()
    assert page.evaluate("!document.querySelector('gym-scene').meshes.has('sensor-rays')")
    scene.get_by_label('Sensor rays', exact=True).check()
    assert page.evaluate("document.querySelector('gym-scene').dataset.rayCount==='2'")
    scene.get_by_label('Heading / control', exact=True).uncheck()
    assert page.evaluate("!document.querySelector('gym-scene').meshes.has('heading-ego')")
    scene.get_by_role('button',name='Whole map',exact=True).click()
    assert not scene.get_by_label('Follow car',exact=True).is_checked()
    scene.get_by_label('Follow car',exact=True).check()
    page.wait_for_function("document.querySelector('gym-scene').follow")
    assert not errors


def test_keyboard_agent_selection_survives_updates_and_reports_native_ids(scene_page):
    page, errors = scene_page
    page.evaluate("""() => {
        const s=document.querySelector('gym-scene');
        s.selectionEvents=[];
        s.addEventListener('gym-agent-selection',e => s.selectionEvents.push(e.detail));
        s.update({...s.pending,vehicles:[...s.pending.vehicles,
          {id:'traffic-one',position:[25,3],heading:1.2,speed:8,controller:'IDM',role:'traffic'}],
          robots:[{id:'carrier',x:38,y:15,direction:1,carrying:true}]});
    }""")
    page.locator('gym-scene canvas').click(position={'x':300,'y':200})
    page.keyboard.press('ArrowRight')
    assert page.locator('gym-scene').get_by_label('Selected agent').input_value() == 'vehicle-traffic-one'
    assert 'traffic-one' in page.locator('.gym-selected-agent').inner_text()
    assert 'Controller: IDM' in page.locator('.gym-selected-agent').inner_text()
    assert 'Speed 8.0 m/s' in page.locator('.gym-selected-agent').inner_text()
    page.evaluate("document.querySelector('gym-scene').update(document.querySelector('gym-scene').pending)")
    assert page.locator('gym-scene').get_by_label('Selected agent').input_value() == 'vehicle-traffic-one'
    page.keyboard.press('ArrowRight')
    assert page.locator('gym-scene').get_by_label('Selected agent').input_value() == 'robot-carrier'
    page.keyboard.press('ArrowLeft')
    event = page.evaluate("document.querySelector('gym-scene').selectionEvents.at(-1)")
    assert event['agent_id'] == 'traffic-one' and event['kind'] == 'vehicle'
    page.evaluate("""() => {const s=document.querySelector('gym-scene');
      s.update({...s.pending,vehicles:s.pending.vehicles.filter(a=>a.id!=='traffic-one')});} """)
    assert page.locator('gym-scene').get_by_label('Selected agent').input_value() == 'robot-carrier'
    page.evaluate("""() => {const s=document.querySelector('gym-scene');
      s.update({...s.pending,vehicles:[],robots:[]});}""")
    count = page.evaluate("document.querySelector('gym-scene').selectionEvents.length")
    page.evaluate("document.querySelector('gym-scene').update(document.querySelector('gym-scene').pending)")
    assert page.evaluate("document.querySelector('gym-scene').selectionEvents.length") == count
    assert 'No agents' in page.locator('.gym-selected-agent').inner_text()
    assert not errors


def test_controls_help_shortcuts_and_input_route_boundaries(scene_page):
    page, errors = scene_page
    scene = page.locator('gym-scene')
    controls = scene.get_by_role('button',name='Controls',exact=True)
    controls.hover()
    help_panel = scene.get_by_role('region',name='Simulation controls')
    help_panel.wait_for()
    assert 'Previous / next agent' in help_panel.inner_text()
    assert 'Play / pause simulation' in help_panel.inner_text()
    controls.focus()
    page.keyboard.press('Escape')
    assert not help_panel.is_visible()
    page.locator('gym-scene canvas').click(position={'x':300,'y':200})
    page.keyboard.press('?')
    assert help_panel.is_visible()
    page.keyboard.press('d')
    assert not page.evaluate("document.querySelector('gym-scene').debugRays")
    scene.get_by_label('Selected agent').focus()
    page.keyboard.press('d')
    assert not page.evaluate("document.querySelector('gym-scene').debugRays")
    scene.get_by_role('button',name='Close controls',exact=True).click()
    page.locator('gym-scene canvas').click(position={'x':300,'y':200})
    page.keyboard.press('m')
    assert not scene.get_by_label('Follow car',exact=True).is_checked()
    page.keyboard.press('f')
    assert scene.get_by_label('Follow car',exact=True).is_checked()
    page.locator('fleet-nav nav a[href="#chat"]').click()
    page.keyboard.press('d')
    assert not page.evaluate("document.querySelector('gym-scene').debugRays")
    assert not errors


def test_driver_camera_button_and_shortcut_use_selected_car(scene_page):
    page, errors = scene_page
    scene = page.locator('gym-scene')
    scene.get_by_role('button', name='Driver view', exact=True).click()
    page.wait_for_function("""() => {
      const s=document.querySelector('gym-scene'), car=s.meshes.get(s.focusKey);
      return s.cameraMode==='driver' && !car.visible &&
        s.camera.position.distanceTo(car.position)<2 && s.camera.fov===72;
    }""")
    assert scene.get_by_role('button', name='Chase view', exact=True).is_visible()
    scene.get_by_role('button', name='Chase view', exact=True).click()
    page.wait_for_function("""() => {
      const s=document.querySelector('gym-scene'), car=s.meshes.get(s.focusKey);
      return car.visible && s.camera.position.distanceTo(car.position)>8 && s.camera.fov===54;
    }""")
    page.locator('gym-scene canvas').focus()
    page.keyboard.press('c')
    page.wait_for_function("document.querySelector('gym-scene').cameraMode==='driver'")
    assert not errors
