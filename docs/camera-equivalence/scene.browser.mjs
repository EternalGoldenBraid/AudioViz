export default async function checkReaderOrbit(page) {
  const check = (condition,message) => { if (!condition) throw new Error(message); };
  const spatial = ["world","homography","parallax","epipolar"];
  const figure = mode => page.locator(`[data-scene-mode="${mode}"]`);
  const points = mode => figure(mode).locator("svg").evaluate(svg =>
    [...svg.querySelectorAll("circle[data-point]")].map(p => [p.dataset.point,p.getAttribute("cx"),p.getAttribute("cy")]));
  const pose = async () => ({
    azimuth: Number(await page.locator("#reader-azimuth").inputValue()),
    elevation: Number(await page.locator("#reader-elevation").inputValue()),
  });
  const drag = async (mode,dx,dy,y=0.75,x=0.3) => {
    const svg = figure(mode).locator("svg");
    await svg.scrollIntoViewIfNeeded();
    const box = await svg.boundingBox();
    const start = {x:box.x+x*box.width,y:box.y+y*box.height};
    await page.mouse.move(start.x,start.y);
    await page.mouse.down();
    await page.mouse.move(start.x+dx,start.y+dy,{steps:12});
    await page.mouse.up();
  };
  await figure("world").locator("svg").waitFor();
  for (const mode of spatial) check(await figure(mode).getByRole("button",{name:"Reset view",exact:true}).count() === 1,
    `${mode} has reader orbit controls`);
  for (const mode of ["flow","predictions"]) check(await figure(mode).getByRole("button",{name:"Reset view",exact:true}).count() === 0,
    `${mode} has no reader orbit controls`);
  const initial = await Promise.all(spatial.map(points));
  const plots = await Promise.all(["flow","predictions"].map(mode => figure(mode).locator("svg").innerHTML()));
  const sceneValues = await Promise.all(["#scene-baseline","#scene-height"].map(id => page.locator(id).inputValue()));
  await drag("parallax",420,130);
  await page.waitForFunction(() => Number(document.querySelector("#reader-azimuth").value) < -150);
  check((await pose()).azimuth === -167,"Dragging wraps the azimuth through 180 degrees");
  check(!await page.getByRole("button",{name:"Comment here",exact:true}).isVisible(),"A drag must not offer a click comment");
  for (let i=0;i<spatial.length;i++) check(JSON.stringify(await points(spatial[i])) !== JSON.stringify(initial[i]),`${spatial[i]} shares the orbit`);
  check(JSON.stringify(await Promise.all(["flow","predictions"].map(mode => figure(mode).locator("svg").innerHTML()))) === JSON.stringify(plots),"Reader motion leaves sensor-coordinate plots unchanged");
  check(JSON.stringify(await Promise.all(["#scene-baseline","#scene-height"].map(id => page.locator(id).inputValue()))) === JSON.stringify(sceneValues),"Orbit leaves the world unchanged");
  await drag("world",0,300,0.15);
  check((await pose()).elevation === -80,"Tilt stops before the camera poles");
  await figure("epipolar").getByRole("button",{name:"Reset view"}).click();
  for (let i=0;i<spatial.length;i++) check(JSON.stringify(await points(spatial[i])) === JSON.stringify(initial[i]),`${spatial[i]} resets to the original projection`);
  await drag("homography",-650,0,0.75,0.9);
  check((await pose()).azimuth === 125,"Reverse dragging wraps below -180 degrees");
  await figure("world").getByRole("button",{name:"Reset view"}).focus();
  await page.keyboard.press("Enter");
  check((await pose()).azimuth === 25 && (await pose()).elevation === 22,"Keyboard reset restores the default viewpoint");
  await figure("parallax").locator("svg text").filter({hasText:/^x₂$/}).click();
  check(await page.getByRole("button",{name:"Comment here",exact:true}).isVisible(),"Clicking a label still offers a comment");
}
